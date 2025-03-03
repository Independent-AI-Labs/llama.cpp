#include "arg.h"
#include "base64.hpp"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "clip.h"
#include "llava.h"
#include "llama.h"
#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef NDEBUG
#include "ggml-alloc.h"
#include "ggml-backend.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

// Function to print debug messages only when debug flag is set
#define DEBUG_PRINT(params, ...) do { if ((params)->debug) { fprintf(stderr, "[DEBUG] "); fprintf(stderr, __VA_ARGS__); } } while (0)


static bool qwen2vl_eval_image_embed(llama_context * ctx_llama, const struct llava_image_embed * image_embed,
                                     int n_batch, int * n_past, int * st_pos_id, struct clip_image_size * image_size,
                                     const common_params* params = nullptr) {
    DEBUG_PRINT(params, "Starting qwen2vl_eval_image_embed with n_batch=%d, n_past=%d, st_pos_id=%d\n",
                n_batch, *n_past, *st_pos_id);
    DEBUG_PRINT(params, "Image size: %dx%d\n", image_size->width, image_size->height);

    int n_embd = llama_model_n_embd(llama_get_model(ctx_llama));

    // Default patch size for standard models
    int patch_size = 28;

    // We need to detect if this is Qwen2.5 based on configuration from the clip model
    bool is_qwen2_5 = false;

    // Get the clip context to check if it's Qwen2.5
    // This would ideally come from the llava_context, but we'll assume Qwen2.5
    // if the image dimensions are correctly resized
    if (image_size->width % 28 == 0 && image_size->height % 28 == 0) {
        is_qwen2_5 = true;
        DEBUG_PRINT(params, "Detected Qwen2.5 VL model based on image dimensions\n");
    }

    // For Qwen2.5, we need to modify patch size calculation
    const int effective_patch_size = is_qwen2_5 ? 14 : patch_size;
    const int merge_size = is_qwen2_5 ? 2 : 1;

    // Calculate grid dimensions based on image size and patch size
    const int grid_h = (image_size->height / effective_patch_size);
    const int grid_w = (image_size->width / effective_patch_size);

    // Calculate number of patches after spatial merging (for Qwen2.5)
    const int merged_grid_h = (grid_h + merge_size - 1) / merge_size;
    const int merged_grid_w = (grid_w + merge_size - 1) / merge_size;

    // Use the correct number of patches based on model type
    const int final_grid_h = is_qwen2_5 ? merged_grid_h : grid_h;
    const int final_grid_w = is_qwen2_5 ? merged_grid_w : grid_w;

    // Account for actual embedded tokens from the model
    // For Qwen2.5, we use the actual number from the embed
    int img_tokens = image_embed->n_image_pos;

    // Verify our calculations match the embedded tokens
    DEBUG_PRINT(params, "Calculated dimensions: grid=%dx%d, merged=%dx%d, final=%dx%d, total=%d, embed_tokens=%d\n",
               grid_w, grid_h, merged_grid_w, merged_grid_h, final_grid_w, final_grid_h,
               final_grid_w * final_grid_h, img_tokens);

    if (params && params->debug) {
        // Extra sanity check for dimensions
        if (is_qwen2_5 && (img_tokens != final_grid_w * final_grid_h)) {
            fprintf(stderr, "[DEBUG] WARNING: Calculated patches (%d) doesn't match embedded tokens (%d)\n",
                    final_grid_w * final_grid_h, img_tokens);
        }
    }

    // Create position vectors - for Qwen2.5 we need 4D positions (MROPE)
    std::vector<llama_pos> mrope_pos;
    mrope_pos.resize(img_tokens * 4, 0); // Initialize with zeros

    DEBUG_PRINT(params, "Setting up MROPE positions for %d patches\n", img_tokens);

    // Initialize base position
    int base_pos = *st_pos_id;

    // Fill in the MROPE positions for each patch
    for (int i = 0; i < img_tokens; i++) {
        int y, x;

        if (is_qwen2_5) {
            // For Qwen2.5: calculate row/col based on merged grid
            y = i / final_grid_w;
            x = i % final_grid_w;
        } else {
            // For standard models: calculate row/col based on original grid
            y = i / grid_w;
            x = i % grid_w;
        }

        // Position 0: Base position (token index)
        mrope_pos[i] = base_pos + i;

        // Position 1: Y position (row)
        mrope_pos[i + img_tokens] = y;

        // Position 2: X position (column)
        mrope_pos[i + img_tokens * 2] = x;

        // Position 3: For Qwen2.5, we use a special value related to the mrope_section
        // From config.json: "mrope_section": [16, 24, 24]
        // Use a different value based on token position/section
        int section_idx = (i < img_tokens/3) ? 0 : ((i < 2*img_tokens/3) ? 1 : 2);
        mrope_pos[i + img_tokens * 3] = section_idx;

        if (params && params->debug && i < 5) {
            DEBUG_PRINT(params, "MROPE pos[%d] = (%d, %d, %d, %d)\n", i,
                       (int)mrope_pos[i], (int)mrope_pos[i + img_tokens],
                       (int)mrope_pos[i + img_tokens * 2], (int)mrope_pos[i + img_tokens * 3]);
        }
    }

    // Update the starting position ID
    *st_pos_id += img_tokens;
    DEBUG_PRINT(params, "Updated st_pos_id to %d\n", *st_pos_id);

    // Process embeddings in batches
    int processed = 0;
    std::vector<llama_pos> batch_mrope_pos;
    batch_mrope_pos.resize(n_batch * 4, 0); // Initialize with zeros

    for (int i = 0; i < img_tokens; i += n_batch) {
        int n_eval = std::min(n_batch, img_tokens - i);

        DEBUG_PRINT(params, "Processing batch %d/%d: %d tokens (total=%d)\n",
                   i/n_batch + 1, (img_tokens + n_batch - 1)/n_batch, n_eval, img_tokens);

        // Copy position data for current batch - be extra careful with memory access
        // First dimension (token indices)
        for (int j = 0; j < n_eval; j++) {
            batch_mrope_pos[j] = mrope_pos[processed + j];
        }

        // Second dimension (y positions)
        for (int j = 0; j < n_eval; j++) {
            batch_mrope_pos[n_batch + j] = mrope_pos[img_tokens + processed + j];
        }

        // Third dimension (x positions)
        for (int j = 0; j < n_eval; j++) {
            batch_mrope_pos[n_batch * 2 + j] = mrope_pos[img_tokens * 2 + processed + j];
        }

        // Fourth dimension (special values)
        for (int j = 0; j < n_eval; j++) {
            batch_mrope_pos[n_batch * 3 + j] = mrope_pos[img_tokens * 3 + processed + j];
        }

        // Create and process the batch
        llama_batch batch = { 0 };
        batch.n_tokens = n_eval;
        batch.token = nullptr;
        batch.embd = image_embed->embed + processed * n_embd;
        batch.pos = batch_mrope_pos.data();
        batch.n_seq_id = nullptr;
        batch.seq_id = nullptr;
        batch.logits = nullptr;

        DEBUG_PRINT(params, "About to call llama_decode with batch size %d\n", n_eval);
        int result = llama_decode(ctx_llama, batch);
        DEBUG_PRINT(params, "llama_decode returned %d\n", result);
        if (result) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }

        *n_past += n_eval;
        processed += n_eval;
        DEBUG_PRINT(params, "Processed %d/%d image tokens, n_past=%d\n", processed, img_tokens, *n_past);
    }

    DEBUG_PRINT(params, "Completed qwen2vl_eval_image_embed successfully\n");
    return true;
}


static bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past, int * st_pos_id, const common_params* params = nullptr) {
    DEBUG_PRINT(params, "Starting eval_tokens with %zu tokens, n_batch=%d, n_past=%d, st_pos_id=%d\n",
                tokens.size(), n_batch, *n_past, *st_pos_id);

    if (params && params->debug && tokens.size() < 10) {
        DEBUG_PRINT(params, "Token IDs: ");
        for (size_t i = 0; i < tokens.size(); i++) {
            fprintf(stderr, "%d ", tokens[i]);
        }
        fprintf(stderr, "\n");
    }

    int N = (int) tokens.size();
    std::vector<llama_pos> pos;
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }

        DEBUG_PRINT(params, "Processing token batch %d/%d: %d tokens\n",
                    i/n_batch + 1, (N + n_batch - 1)/n_batch, n_eval);

        auto batch = llama_batch_get_one(&tokens[i], n_eval);

        // Set up position IDs for MROPE (4D positions)
        pos.resize(batch.n_tokens * 4);
        std::fill(pos.begin(), pos.end(), 0);

        // First dimension: sequential position
        for (int j = 0; j < batch.n_tokens; j++) {
            pos[j] = *st_pos_id + j;
        }

        // Second dimension: also sequential position (for compatibility with Qwen2.5 models)
        for (int j = 0; j < batch.n_tokens; j++) {
            pos[batch.n_tokens + j] = *st_pos_id + j;
        }

        // Third and fourth dimensions: set to zero for text tokens
        std::fill(pos.begin() + batch.n_tokens * 2, pos.end(), 0);

        batch.pos = pos.data();

        DEBUG_PRINT(params, "Calling llama_decode for batch with %d tokens\n", n_eval);
        if (llama_decode(ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
        *st_pos_id += n_eval;

        DEBUG_PRINT(params, "Updated n_past=%d, st_pos_id=%d\n", *n_past, *st_pos_id);
    }

    DEBUG_PRINT(params, "Completed eval_tokens successfully\n");
    return true;
}

static bool eval_id(struct llama_context * ctx_llama, int id, int * n_past, int * st_pos_id, const common_params* params = nullptr) {
    DEBUG_PRINT(params, "eval_id: Processing token ID %d, n_past=%d, st_pos_id=%d\n", id, *n_past, *st_pos_id);

    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past, st_pos_id, params);
}

static bool eval_string(struct llama_context * ctx_llama, const char* str, int n_batch, int * n_past, int * st_pos_id, bool add_bos, const common_params* params = nullptr){
    DEBUG_PRINT(params, "eval_string: Processing string \"%s\", n_batch=%d, n_past=%d, st_pos_id=%d, add_bos=%d\n",
                str, n_batch, *n_past, *st_pos_id, add_bos);

    std::string              str2     = str;
    std::vector<llama_token> embd_inp = common_tokenize(ctx_llama, str2, add_bos, true);

    DEBUG_PRINT(params, "Tokenized to %zu tokens\n", embd_inp.size());

    if (params && params->debug && embd_inp.size() < 20) {
        DEBUG_PRINT(params, "Token IDs: ");
        for (size_t i = 0; i < embd_inp.size(); i++) {
            fprintf(stderr, "%d ", embd_inp[i]);
        }
        fprintf(stderr, "\n");

        DEBUG_PRINT(params, "Token pieces: ");
        for (size_t i = 0; i < embd_inp.size(); i++) {
            fprintf(stderr, "\"%s\" ", common_token_to_piece(ctx_llama, embd_inp[i]).c_str());
        }
        fprintf(stderr, "\n");
    }

    eval_tokens(ctx_llama, embd_inp, n_batch, n_past, st_pos_id, params);
    return true;
}

static const char * sample(struct common_sampler * smpl,
                           struct llama_context * ctx_llama,
                           int * n_past, int * st_pos_id,
                           const common_params* params = nullptr) {
    DEBUG_PRINT(params, "sample: n_past=%d, st_pos_id=%d\n", *n_past, *st_pos_id);

    const llama_token id = common_sampler_sample(smpl, ctx_llama, -1);
    DEBUG_PRINT(params, "Sampled token ID: %d\n", id);

    common_sampler_accept(smpl, id, true);

    const llama_model * model = llama_get_model(ctx_llama);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    static std::string ret;
    if (llama_vocab_is_eog(vocab, id)) {
        ret = "</s>";
        DEBUG_PRINT(params, "Sampled EOG token (end of generation)\n");
    } else {
        ret = common_token_to_piece(ctx_llama, id);
        DEBUG_PRINT(params, "Token text: \"%s\"\n", ret.c_str());
    }
    eval_id(ctx_llama, id, n_past, st_pos_id, params);
    return ret.c_str();
}

static const char* IMG_BASE64_TAG_BEGIN = "<img src=\"data:image/jpeg;base64,";
static const char* IMG_BASE64_TAG_END = "\">";

static void find_image_tag_in_prompt(const std::string& prompt, size_t& begin_out, size_t& end_out) {
    begin_out = prompt.find(IMG_BASE64_TAG_BEGIN);
    end_out = prompt.find(IMG_BASE64_TAG_END, (begin_out == std::string::npos) ? 0UL : begin_out);
}

static bool prompt_contains_image(const std::string& prompt) {
    size_t begin, end;
    find_image_tag_in_prompt(prompt, begin, end);
    return (begin != std::string::npos);
}

// replaces the base64 image tag in the prompt with `replacement`
static llava_image_embed * llava_image_embed_make_with_prompt_base64(struct clip_ctx * ctx_clip, int n_threads,
                                                                     const std::string& prompt, const common_params* params = nullptr) {
    DEBUG_PRINT(params, "llava_image_embed_make_with_prompt_base64: Processing base64 image in prompt\n");

    size_t img_base64_str_start, img_base64_str_end;
    find_image_tag_in_prompt(prompt, img_base64_str_start, img_base64_str_end);
    if (img_base64_str_start == std::string::npos || img_base64_str_end == std::string::npos) {
        LOG_ERR("%s: invalid base64 image tag. must be %s<base64 byte string>%s\n", __func__, IMG_BASE64_TAG_BEGIN, IMG_BASE64_TAG_END);
        return NULL;
    }

    auto base64_bytes_start = img_base64_str_start + strlen(IMG_BASE64_TAG_BEGIN);
    auto base64_bytes_count = img_base64_str_end - base64_bytes_start;
    auto base64_str = prompt.substr(base64_bytes_start, base64_bytes_count);

    DEBUG_PRINT(params, "Base64 image string length: %zu bytes\n", base64_str.length());

    auto required_bytes = base64::required_encode_size(base64_str.size());
    auto img_bytes = std::vector<unsigned char>(required_bytes);
    base64::decode(base64_str.begin(), base64_str.end(), img_bytes.begin());

    DEBUG_PRINT(params, "Decoded image size: %zu bytes\n", img_bytes.size());

    auto embed = llava_image_embed_make_with_bytes(ctx_clip, n_threads, img_bytes.data(), img_bytes.size());
    if (!embed) {
        LOG_ERR("%s: could not load image from base64 string.\n", __func__);
        return NULL;
    }

    DEBUG_PRINT(params, "Successfully created image embed with %d patches\n", embed->n_image_pos);
    return embed;
}

static std::string remove_image_from_prompt(const std::string& prompt, const char * replacement = "") {
    size_t begin, end;
    find_image_tag_in_prompt(prompt, begin, end);
    if (begin == std::string::npos || end == std::string::npos) {
        return prompt;
    }
    auto pre = prompt.substr(0, begin);
    auto post = prompt.substr(end + strlen(IMG_BASE64_TAG_END));
    return pre + replacement + post;
}

struct llava_context {
    struct clip_ctx * ctx_clip = NULL;
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
};

static void print_usage(int, char ** argv) {
    LOG("\n example usage:\n");
    LOG("\n     %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf> --image <path/to/an/image.jpg> --image <path/to/another/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"] [--debug]\n", argv[0]);
    LOG("\n note: a lower temperature value like 0.1 is recommended for better quality.\n");
    LOG("\n optional flags:\n");
    LOG("  --debug                 Enable debug output\n");
}

static struct llava_image_embed * load_image(llava_context * ctx_llava, common_params * params, const std::string & fname) {
    DEBUG_PRINT(params, "load_image: Loading image from %s\n",
                fname.empty() ? "base64 in prompt" : fname.c_str());

    // load and preprocess the image
    llava_image_embed * embed = NULL;
    auto prompt = params->prompt;
    if (prompt_contains_image(prompt)) {
        if (!params->image.empty()) {
            LOG_INF("using base64 encoded image instead of command line image path\n");
        }
        DEBUG_PRINT(params, "Extracting image from base64 in prompt\n");
        embed = llava_image_embed_make_with_prompt_base64(ctx_llava->ctx_clip, params->cpuparams.n_threads, prompt, params);
        if (!embed) {
            LOG_ERR("%s: can't load image from prompt\n", __func__);
            return NULL;
        }
        params->prompt = remove_image_from_prompt(prompt);
        DEBUG_PRINT(params, "Prompt after removing image: %s\n", params->prompt.c_str());
    } else {
        DEBUG_PRINT(params, "Loading image from file: %s\n", fname.c_str());
        embed = llava_image_embed_make_with_filename(ctx_llava->ctx_clip, params->cpuparams.n_threads, fname.c_str());
        if (!embed) {
            fprintf(stderr, "%s: is %s really an image file?\n", __func__, fname.c_str());
            return NULL;
        }
    }

    DEBUG_PRINT(params, "Image loaded successfully with %d patches\n", embed->n_image_pos);
    return embed;
}

void adjust_qwen2_sampling_params(struct llava_context* ctx_llava, common_params* params) {
    if (clip_is_qwen2_5vl(ctx_llava->ctx_clip)) {
        DEBUG_PRINT(params, "Adjusting sampling parameters for Qwen2.5 VL\n");

        // Only modify if user hasn't explicitly set these parameters
        if (params->sampling.temp == 0.8f) {  // Default value
            params->sampling.temp = 0.7f;     // Slightly lower temperature for Qwen2.5
            DEBUG_PRINT(params, "Set temperature to %f for Qwen2.5\n", params->sampling.temp);
        }

        if (params->sampling.top_p == 0.95f) {  // Default value
            params->sampling.top_p = 0.9f;      // Slightly lower top_p
            DEBUG_PRINT(params, "Set top_p to %f for Qwen2.5\n", params->sampling.top_p);
        }

        // Set a reasonable max length if not specified
        if (params->n_predict < 0) {
            params->n_predict = 512;
            DEBUG_PRINT(params, "Set n_predict to %d for Qwen2.5\n", params->n_predict);
        }
    }
}

static void process_prompt(struct llava_context * ctx_llava, struct llava_image_embed * image_embed, common_params * params, const std::string & prompt) {
    DEBUG_PRINT(params, "process_prompt: Processing prompt: %s\n", prompt.c_str());
    DEBUG_PRINT(params, "Image embed: %p, n_image_pos: %d\n",
                (void*)image_embed, image_embed ? image_embed->n_image_pos : 0);

    int n_past = 0;
    int cur_pos_id = 0;

    const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;
    DEBUG_PRINT(params, "Max target length: %d\n", max_tgt_len);

    // Check if we're dealing with Qwen2.5 VL
    bool is_qwen2_5 = false;
    if (ctx_llava && ctx_llava->ctx_clip) {
        is_qwen2_5 = clip_is_qwen2_5vl(ctx_llava->ctx_clip);
        DEBUG_PRINT(params, "Model detection: is_qwen2_5 = %d\n", is_qwen2_5);
    }

    // Format the prompt according to model requirements
    std::string system_prompt, user_prompt;

    if (is_qwen2_5) {
        // Qwen2.5 VL specific format based on the official chat template
        // From chat_template.json in the model repo
        system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n";

        // For Qwen2.5, we need to check if the image is already mentioned in the prompt
        if (prompt.find("<|vision_start|>") != std::string::npos) {
            // The prompt already contains vision tags, use it as is
            user_prompt = prompt + "<|im_end|>\n<|im_start|>assistant\n";
            DEBUG_PRINT(params, "Using user-provided prompt with vision tags\n");
        } else {
            // Add vision tags for the image
            user_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
            DEBUG_PRINT(params, "Using Qwen2.5 VL format with auto-added vision tags\n");
        }
    } else {
        // Try to detect if prompt already has a specific format
        size_t image_pos = prompt.find("<|vision_start|>");
        if (image_pos != std::string::npos) {
            // New templating mode: Provide the full prompt including system message
            system_prompt = prompt.substr(0, image_pos);
            user_prompt = prompt.substr(image_pos);
            DEBUG_PRINT(params, "Using templating mode with <|vision_start|>\n");
        } else {
            // Default llava-1.5 native mode
            system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>";
            user_prompt = "<|vision_end|>" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
            DEBUG_PRINT(params, "Using llava-1.5 native mode\n");
        }
    }

    LOG_INF("system_prompt: %s\n", system_prompt.c_str());
    if (params->verbose_prompt) {
        auto tmp = common_tokenize(ctx_llava->ctx_llama, system_prompt, true, true);
        for (int i = 0; i < (int) tmp.size(); i++) {
            LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
        }
    }

    LOG_INF("user_prompt: %s\n", user_prompt.c_str());
    if (params->verbose_prompt) {
        auto tmp = common_tokenize(ctx_llava->ctx_llama, user_prompt, true, true);
        for (int i = 0; i < (int) tmp.size(); i++) {
            LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
        }
    }

    DEBUG_PRINT(params, "Evaluating system prompt: %s\n", system_prompt.c_str());
    eval_string(ctx_llava->ctx_llama, system_prompt.c_str(), params->n_batch, &n_past, &cur_pos_id, true, params);

    // For Qwen2.5, determine if we need to inject the image embed based on vision tags
    bool inject_image = true;
    if (is_qwen2_5) {
        // Check if the user prompt already has vision tags
        bool has_vision_tags = (user_prompt.find("<|vision_start|>") != std::string::npos);

        // If we have vision tags and image embed, inject at the right position
        if (has_vision_tags && image_embed != nullptr) {
            // For Qwen2.5, we need special handling based on the tags
            size_t vision_tag_pos = user_prompt.find("<|image_pad|>");
            if (vision_tag_pos != std::string::npos) {
                // Find the tag positions
                size_t before_tag = user_prompt.find("<|vision_start|>");
                size_t after_tag = user_prompt.find("<|vision_end|>", vision_tag_pos);

                // Split the prompt around the image
                std::string before_img = user_prompt.substr(0, vision_tag_pos + std::string("<|image_pad|>").length());
                std::string after_img = user_prompt.substr(after_tag);

                // First evaluate text before image
                DEBUG_PRINT(params, "Evaluating text before image: %s\n", before_img.c_str());
                eval_string(ctx_llava->ctx_llama, before_img.c_str(), params->n_batch, &n_past, &cur_pos_id, false, params);

                // Then evaluate image embed
                DEBUG_PRINT(params, "Evaluating image embed with %d patches\n", image_embed->n_image_pos);
                auto image_size = clip_get_load_image_size(ctx_llava->ctx_clip);
                qwen2vl_eval_image_embed(ctx_llava->ctx_llama, image_embed, params->n_batch, &n_past, &cur_pos_id, image_size, params);

                // Finally evaluate text after image
                DEBUG_PRINT(params, "Evaluating text after image: %s\n", after_img.c_str());
                eval_string(ctx_llava->ctx_llama, after_img.c_str(), params->n_batch, &n_past, &cur_pos_id, false, params);

                // We already handled the image embedding
                inject_image = false;
            }
        }
    }

    // Standard image injection if needed
    if (inject_image && image_embed != nullptr) {
        DEBUG_PRINT(params, "Evaluating image embed with %d patches\n", image_embed->n_image_pos);
        auto image_size = clip_get_load_image_size(ctx_llava->ctx_clip);
        qwen2vl_eval_image_embed(ctx_llava->ctx_llama, image_embed, params->n_batch, &n_past, &cur_pos_id, image_size, params);
    }

    // If we haven't already processed the user prompt (in the Qwen2.5 case)
    if (inject_image) {
        DEBUG_PRINT(params, "Evaluating user prompt: %s\n", user_prompt.c_str());
        eval_string(ctx_llava->ctx_llama, user_prompt.c_str(), params->n_batch, &n_past, &cur_pos_id, false, params);
    }

    // generate the response
    DEBUG_PRINT(params, "Starting response generation (max_length=%d)\n", max_tgt_len);

    LOG("\n");

    struct common_sampler * smpl = common_sampler_init(ctx_llava->model, params->sampling);

    // Adjust sampling parameters for Qwen2.5
    if (is_qwen2_5) {
        // Only modify if user hasn't explicitly set these parameters
        if (params->sampling.temp == 0.8f) {  // Default value
            params->sampling.temp = 0.7f;     // Slightly lower temperature for Qwen2.5
            DEBUG_PRINT(params, "Set temperature to %f for Qwen2.5\n", params->sampling.temp);
        }

        if (params->sampling.top_p == 0.95f) {  // Default value
            params->sampling.top_p = 0.9f;      // Slightly lower top_p
            DEBUG_PRINT(params, "Set top_p to %f for Qwen2.5\n", params->sampling.top_p);
        }
    }

    if (!smpl) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
        exit(1);
    }

    std::string response = "";

    for (int i = 0; i < max_tgt_len; i++) {
        DEBUG_PRINT(params, "Generating token %d/%d\n", i+1, max_tgt_len);
        const char * tmp = sample(smpl, ctx_llava->ctx_llama, &n_past, &cur_pos_id, params);
        response += tmp;
        if (strcmp(tmp, "</s>") == 0) {
            DEBUG_PRINT(params, "End of sequence token generated\n");
            break;
        }
        if (strstr(tmp, "###")) {
            DEBUG_PRINT(params, "Found '###' terminator (Yi-VL behavior)\n");
            break; // Yi-VL behavior
        }
        LOG("%s", tmp);
        if (strstr(response.c_str(), "<|im_end|>")) {
            DEBUG_PRINT(params, "Found '<|im_end|>' terminator\n");
            break; // Common terminator for chat models
        }
        if (strstr(response.c_str(), "<|im_start|>")) {
            DEBUG_PRINT(params, "Found '<|im_start|>' terminator\n");
            break; // Start of next turn
        }
        // Specific Qwen2.5 terminators from the chat template
        if (is_qwen2_5 &&
            (strstr(response.c_str(), "<|im_end|>") ||
             strstr(response.c_str(), "<|im_start|>"))) {
            DEBUG_PRINT(params, "Found Qwen2.5 specific terminator based on chat template\n");
            break;
        }
        if (strstr(response.c_str(), "USER:")) {
            DEBUG_PRINT(params, "Found 'USER:' terminator (mistral llava-1.6)\n");
            break; // mistral llava-1.6
        }

        fflush(stdout);
    }

    DEBUG_PRINT(params, "Generation complete. Response length: %zu characters\n", response.length());

    common_sampler_free(smpl);
    LOG("\n");
}

static struct llama_model * llava_init(common_params * params) {
    DEBUG_PRINT(params, "llava_init: Initializing LLAMA model from %s\n", params->model.c_str());

    llama_backend_init();
    llama_numa_init(params->numa);

    llama_model_params model_params = common_model_params_to_llama(*params);

    DEBUG_PRINT(params, "Loading model from %s\n", params->model.c_str());
    llama_model * model = llama_model_load_from_file(params->model.c_str(), model_params);
    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n" , __func__);
        return NULL;
    }

    DEBUG_PRINT(params, "Model loaded successfully\n");

    return model;
}

static struct llava_context * llava_init_context(common_params * params, llama_model * model) {
    const char * clip_path = params->mmproj.c_str();
    DEBUG_PRINT(params, "llava_init_context: Initializing LLAVA context with CLIP from %s\n", clip_path);

    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
        DEBUG_PRINT(params, "Using default prompt: %s\n", prompt.c_str());
    }

    DEBUG_PRINT(params, "Loading CLIP model\n");
    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/ 1);
    DEBUG_PRINT(params, "CLIP model loaded successfully\n");

    llama_context_params ctx_params = common_context_params_to_llama(*params);
    ctx_params.n_ctx = params->n_ctx < 2048 ? 2048 : params->n_ctx; // we need a longer context size to process image embeddings

    DEBUG_PRINT(params, "Initializing LLAMA context\n");
    llama_context * ctx_llama = llama_init_from_model(model, ctx_params);

    if (ctx_llama == NULL) {
        LOG_ERR("%s: failed to create the llama_context\n" , __func__);
        return NULL;
    }
    DEBUG_PRINT(params, "LLAMA context initialized successfully\n");

    auto * ctx_llava = (struct llava_context *)malloc(sizeof(llava_context));

    ctx_llava->ctx_llama = ctx_llama;
    ctx_llava->ctx_clip = ctx_clip;
    ctx_llava->model = model;

    DEBUG_PRINT(params, "LLAVA context created successfully\n");
    return ctx_llava;
}

static void llava_free(struct llava_context * ctx_llava) {
    if (ctx_llava->ctx_clip) {
        clip_free(ctx_llava->ctx_clip);
        ctx_llava->ctx_clip = NULL;
    }

    llama_free(ctx_llava->ctx_llama);
    llama_model_free(ctx_llava->model);
    llama_backend_free();
}

#ifndef NDEBUG
//
//static void debug_test_mrope_2d() {
//    fprintf(stderr, "[DEBUG] debug_test_mrope_2d: Starting MROPE 2D test\n");
//
//    // 1. Initialize backend
//    ggml_backend_t backend = NULL;
//    std::string backend_name = "";
//#ifdef GGML_USE_CUDA
//    fprintf(stderr, "[DEBUG] Using CUDA backend\n");
//    backend = ggml_backend_cuda_init(0); // init device 0
//    backend_name = "cuda";
//    if (!backend) {
//        fprintf(stderr, "[DEBUG] ggml_backend_cuda_init() failed\n");
//    }
//#endif
//    // if there aren't GPU Backends fallback to CPU backend
//    if (!backend) {
//        backend = ggml_backend_cpu_init();
//        backend_name = "cpu";
//        fprintf(stderr, "[DEBUG] Using CPU backend\n");
//    }
//
//    // Calculate the size needed to allocate
//    size_t ctx_size = 0;
//    ctx_size += 2 * ggml_tensor_overhead(); // tensors
//    // no need to allocate anything else!
//
//    // 2. Allocate `ggml_context` to store tensor data
//    struct ggml_init_params params = {
//        /*.mem_size   =*/ ctx_size,
//        /*.mem_buffer =*/ NULL,
//        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_backend_alloc_ctx_tensors()
//    };
//    struct ggml_context * ctx = ggml_init(params);
//
//    struct ggml_tensor * inp_raw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 12, 30);
//    ggml_set_name(inp_raw, "inp_raw");
//    ggml_set_input(inp_raw);
//
//    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 30 * 4);
//    ggml_set_name(pos, "pos");
//    ggml_set_input(pos);
//
//    fprintf(stderr, "[DEBUG] Created tensors inp_raw: [%d, %d, %d], pos: [%d]\n",
//            (int)inp_raw->ne[0], (int)inp_raw->ne[1], (int)inp_raw->ne[2], (int)pos->ne[0]);
//
//    std::vector<float> dummy_q;
//    dummy_q.resize(128 * 12 * 30);
//    std::fill(dummy_q.begin(), dummy_q.end(), 0.1);
//    // memcpy(inp_raw->data, dummy_q.data(), 128 * 12 * 30 * ggml_element_size(inp_raw));
//
//    std::vector<int> pos_id;
//    pos_id.resize(30 * 4);
//    for (int i = 0; i < 30; i ++) {
//        pos_id[i] = i;
//        pos_id[i + 30] = i + 10;
//        pos_id[i + 60] = i + 20;
//        pos_id[i + 90] = i + 30;
//    }
//    int sections[4] = {32, 32, 0, 0};
//
//    fprintf(stderr, "[DEBUG] First 5 position IDs: [%d, %d, %d, %d, %d]\n",
//            pos_id[0], pos_id[1], pos_id[2], pos_id[3], pos_id[4]);
//
//    // 4. Allocate a `ggml_backend_buffer` to store all tensors
//    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
//    fprintf(stderr, "[DEBUG] Allocated backend buffer for tensors\n");
//
//    // 5. Copy tensor data from main memory (RAM) to backend buffer
//    ggml_backend_tensor_set(inp_raw, dummy_q.data(), 0, ggml_nbytes(inp_raw));
//    ggml_backend_tensor_set(pos, pos_id.data(), 0, ggml_nbytes(pos));
//    fprintf(stderr, "[DEBUG] Copied tensor data to backend\n");
//
//    // 6. Create a `ggml_cgraph` for mul_mat operation
//    struct ggml_cgraph * gf = NULL;
//    struct ggml_context * ctx_cgraph = NULL;
//
//    // create a temporally context to build the graph
//    struct ggml_init_params params0 = {
//        /*.mem_size   =*/ ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
//        /*.mem_buffer =*/ NULL,
//        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
//    };
//    ctx_cgraph = ggml_init(params0);
//    gf = ggml_new_graph(ctx_cgraph);
//    fprintf(stderr, "[DEBUG] Created computational graph\n");
//
//    struct ggml_tensor * result0 = ggml_rope_multi(
//        ctx_cgraph, inp_raw, pos, nullptr,
//        128/2, sections, LLAMA_ROPE_TYPE_VISION, 32768, 1000000, 1,
//        0, 1, 32, 1);
//    fprintf(stderr, "[DEBUG] Added ggml_rope_multi operation to graph\n");
//
//    // Add "result" tensor and all of its dependencies to the cgraph
//    ggml_build_forward_expand(gf, result0);
//    fprintf(stderr, "[DEBUG] Built forward graph with %d nodes\n", gf->n_nodes);
//
//    // 7. Create a `ggml_gallocr` for cgraph computation
//    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
//    ggml_gallocr_alloc_graph(allocr, gf);
//    fprintf(stderr, "[DEBUG] Allocated memory for computation\n");
//
//    // 9. Run the computation
//    int n_threads = 1; // Optional: number of threads to perform some operations with multi-threading
//    if (ggml_backend_is_cpu(backend)) {
//        ggml_backend_cpu_set_n_threads(backend, n_threads);
//    }
//    fprintf(stderr, "[DEBUG] Computing graph with %d threads\n", n_threads);
//    ggml_backend_graph_compute(backend, gf);
//    fprintf(stderr, "[DEBUG] Computation complete\n");
//
//    // 10. Retrieve results (output tensors)
//    // in this example, output tensor is always the last tensor in the graph
//    struct ggml_tensor * result = result0;
//    // struct ggml_tensor * result = gf->nodes[gf->n_nodes - 1];
//    float * result_data = (float *)malloc(ggml_nbytes(result));
//    // because the tensor data is stored in device buffer, we need to copy it back to RAM
//    ggml_backend_tensor_get(result, result_data, 0, ggml_nbytes(result));
//    const std::string bin_file = "mrope_2d_" + backend_name +".bin";
//    std::ofstream outFile(bin_file, std::ios::binary);
//
//    fprintf(stderr, "[DEBUG] Result tensor shape: [%d, %d, %d, %d]\n",
//            (int)result->ne[0], (int)result->ne[1], (int)result->ne[2], (int)result->ne[3]);
//
//    if (outFile.is_open()) {
//        outFile.write(reinterpret_cast<const char*>(result_data), ggml_nbytes(result));
//        outFile.close();
//        std::cout << "Data successfully written to " + bin_file << std::endl;
//        fprintf(stderr, "[DEBUG] Saved result to %s (%zu bytes)\n", bin_file.c_str(), ggml_nbytes(result));
//    } else {
//        std::cerr << "Error opening file!" << std::endl;
//        fprintf(stderr, "[DEBUG] Failed to save result to %s\n", bin_file.c_str());
//    }
//
//    free(result_data);
//    // 11. Free memory and exit
//    ggml_free(ctx_cgraph);
//    ggml_gallocr_free(allocr);
//    ggml_free(ctx);
//    ggml_backend_buffer_free(buffer);
//    ggml_backend_free(backend);
//    fprintf(stderr, "[DEBUG] Cleaned up resources\n");
//}

static void debug_dump_img_embed(struct llava_context * ctx_llava) {
    fprintf(stderr, "[DEBUG] debug_dump_img_embed: Starting image embedding dump\n");

    int n_embd  = llama_model_n_embd(llama_get_model(ctx_llava->ctx_llama));
    int ne = n_embd * 4;
    fprintf(stderr, "[DEBUG] Embedding dimension: %d, total size: %d\n", n_embd, ne);

    float vals[56 * 56 * 3];
    // float embd[ne];
    std::vector<float> embd;
    embd.resize(ne);

    fprintf(stderr, "[DEBUG] Generating test image data (56x56 gradient)\n");
    for (int i = 0; i < 56*56; i++)
    {
        for (int c = 0; c < 3; c++)
            vals[i * 3 + c] = (float)(i % (56 * 56)) / (56*56);
    }

    fprintf(stderr, "[DEBUG] Encoding test image\n");
    clip_encode_float_image(ctx_llava->ctx_clip, 16, vals, 56, 56, embd.data());
    fprintf(stderr, "[DEBUG] Image encoded successfully\n");

    fprintf(stderr, "[DEBUG] Saving embedding to img_embed.bin\n");
    std::ofstream outFile("img_embed.bin", std::ios::binary);
    if (outFile.is_open()) {
        outFile.write(reinterpret_cast<const char*>(embd.data()), ne * sizeof(float));

        outFile.close();
        std::cout << "Data successfully written to img_embed.bin" << std::endl;
        fprintf(stderr, "[DEBUG] Embedding saved successfully (%d bytes)\n", ne * (int)sizeof(float));
    } else {
        std::cerr << "Error opening file!" << std::endl;
        fprintf(stderr, "[DEBUG] Failed to save embedding\n");
    }
}

#endif


int main(int argc, char ** argv) {
    ggml_time_init();

    common_params params;

    // Add a debug flag option to common_params parser
    // In common.cpp, add:
    // gpt_params.add_argument("--debug", "-D", "Enable debug output", false);
    // And in common.h:
    // bool debug = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LLAVA, print_usage)) {
        return 1;
    }

    // If debug mode is enabled, print it out
    if (params.debug) {
        fprintf(stderr, "[DEBUG] Debug mode enabled\n");
    }

    common_init();
    DEBUG_PRINT(&params, "qwen2vl-cli initialized\n");

    if (params.mmproj.empty() || (params.image.empty() && !prompt_contains_image(params.prompt))) {
        print_usage(argc, argv);
        return 1;
    }

    DEBUG_PRINT(&params, "Initializing LLAVA model\n");
    auto * model = llava_init(&params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to init llava model\n", __func__);
        return 1;
    }
    DEBUG_PRINT(&params, "LLAVA model initialized successfully\n");

    if (prompt_contains_image(params.prompt)) {
        DEBUG_PRINT(&params, "Using image from base64 in prompt\n");
        auto * ctx_llava = llava_init_context(&params, model);

        auto * image_embed = load_image(ctx_llava, &params, "");

        // process the prompt
        process_prompt(ctx_llava, image_embed, &params, params.prompt);

        llama_perf_context_print(ctx_llava->ctx_llama);
        llava_image_embed_free(image_embed);
        ctx_llava->model = NULL;
        llava_free(ctx_llava);
#ifndef NDEBUG
    } else if (params.image[0].empty()) {
        DEBUG_PRINT(&params, "Running debug tests (no image provided)\n");
        auto ctx_llava = llava_init_context(&params, model);

//        debug_test_mrope_2d();
        debug_dump_img_embed(ctx_llava);

        llama_perf_context_print(ctx_llava->ctx_llama);
        ctx_llava->model = NULL;
        llava_free(ctx_llava);
#endif
    } else {
        DEBUG_PRINT(&params, "Processing %zu images\n", params.image.size());
        for (auto & image : params.image) {
            DEBUG_PRINT(&params, "Processing image: %s\n", image.c_str());
            auto * ctx_llava = llava_init_context(&params, model);

            auto * image_embed = load_image(ctx_llava, &params, image);
            if (!image_embed) {
                LOG_ERR("%s: failed to load image %s. Terminating\n\n", __func__, image.c_str());
                return 1;
            }

            // process the prompt
            process_prompt(ctx_llava, image_embed, &params, params.prompt);

            llama_perf_context_print(ctx_llava->ctx_llama);
            llava_image_embed_free(image_embed);
            ctx_llava->model = NULL;
            llava_free(ctx_llava);
        }
    }

    llama_model_free(model);
    DEBUG_PRINT(&params, "Program completed successfully\n");

    return 0;
}
