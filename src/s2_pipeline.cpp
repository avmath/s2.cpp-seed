#include "../include/s2_pipeline.h"
#include <cstdio>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace s2 {

static void safe_print_ln(const std::string& msg) {
    fputs(msg.c_str(), stdout);
    fputc('\n', stdout);
}

static void safe_print_error_ln(const std::string& msg) {
    fputs(msg.c_str(), stderr);
    fputc('\n', stderr);
}


static std::string make_memory_audio_cache_key(const void* data, size_t size) {
    const unsigned char* bytes = static_cast<const unsigned char*>(data);
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= 1099511628211ull;
    }

    std::ostringstream os;
    os << "memory:" << size << ':' << std::hex << std::setw(16) << std::setfill('0') << hash;
    return os.str();
}

static bool cache_matches(const Pipeline::PermanentPromptCache& cache,
                          const std::string& audio_key,
                          const std::string& prompt_text) {
    return cache.valid &&
           cache.audio_key == audio_key &&
           cache.prompt_text == prompt_text &&
           cache.t_prompt > 0 &&
           !cache.ref_codes.empty();
}

Pipeline::Pipeline() {}
Pipeline::~Pipeline() {}

bool Pipeline::init(const PipelineParams & params) {
    safe_print_ln("--- Pipeline Init ---");

    if (!tokenizer_.load(params.tokenizer_path)) {
        safe_print_error_ln("Pipeline error: could not load tokenizer from " + params.tokenizer_path);
        return false;
    }

    if (!model_.load(params.model_path, params.gpu_device, params.backend_type)) {
        safe_print_error_ln("Pipeline error: could not load model from " + params.model_path);
        return false;
    }

    if (!codec_.load(params.model_path, -1, -1)) {
        safe_print_error_ln("Pipeline error: could not load codec from " + params.model_path);
        return false;
    }

    {
        const ModelHParams & hp = model_.hparams();
        TokenizerConfig & tc    = tokenizer_.config();
        if (hp.semantic_begin_id > 0) tc.semantic_begin_id = hp.semantic_begin_id;
        if (hp.semantic_end_id   > 0) tc.semantic_end_id   = hp.semantic_end_id;
        if (hp.num_codebooks     > 0) tc.num_codebooks     = hp.num_codebooks;
        if (hp.codebook_size     > 0) tc.codebook_size     = hp.codebook_size;
        if (hp.vocab_size        > 0) tc.vocab_size        = hp.vocab_size;
    }

    initialized_ = true;
    return true;
}

bool Pipeline::synthesize(const PipelineParams & params) {
    std::vector<float> audio_out;
    AudioData ref_audio;

    if (!params.prompt_audio_path.empty() && params.prompt_text.empty()) {
        safe_print_error_ln("Pipeline error: prompt audio was provided without prompt text.");
        return false;
    }

    PipelineParams effective_params = params;
    if (effective_params.prompt_audio_cache_key.empty() && !effective_params.prompt_audio_path.empty()) {
        effective_params.prompt_audio_cache_key = "path:" + effective_params.prompt_audio_path;
    }

    if (!effective_params.prompt_audio_path.empty()) {
        bool use_cached_prompt = false;
        if (effective_params.permanent_prompt) {
            std::lock_guard<std::mutex> cache_lock(permanent_prompt_cache_mutex_);
            use_cached_prompt = cache_matches(permanent_prompt_cache_, effective_params.prompt_audio_cache_key, effective_params.prompt_text);
        }
        if (use_cached_prompt) {
            safe_print_ln("Using cached permanent reference audio: " + effective_params.prompt_audio_path);
        } else {
            safe_print_ln("Loading reference audio: " + effective_params.prompt_audio_path);
            if (!load_audio(effective_params.prompt_audio_path, ref_audio, codec_.sample_rate())) {
                safe_print_error_ln("Pipeline warning: load_audio failed, running without reference audio.");
            }
        }
    }
    int32_t AudioOutFrames = 0;
    if (!this->synthesize_raw(effective_params, ref_audio, audio_out, &AudioOutFrames)) {
        safe_print_error_ln("Pipeline error: synthesis failed.");
        return false;
    }

    if (params.trim_silence && !audio_out.empty()) {
        auto trimmed = audio_trim_trailing_silence(audio_out.data(), audio_out.size(), codec_.sample_rate());
        if (!trimmed.empty()) audio_out = std::move(trimmed);
    }

    if (params.normalize_dynamic && !audio_out.empty()) {
        audio_out = audio_normalize_dynamic(audio_out.data(), audio_out.size(), codec_.sample_rate());
    }

    if (!save_audio(params.output_path, audio_out, codec_.sample_rate(),
                    false, params.normalize_output)) {
        safe_print_error_ln("Pipeline error: save_audio failed to " + params.output_path);
        return false;
    }

    safe_print_ln("Saved audio to: " + params.output_path);
    return true;
}

bool Pipeline::synthesize_to_memory(const PipelineParams & params, void** ref_audio_buffer, size_t* ref_audio_size, void** wav_buffer, size_t* wav_size) {
    std::vector<float> audio_out;
    AudioData ref_audio;

    if (ref_audio_buffer && ref_audio_size && *ref_audio_buffer && *ref_audio_size > 0 && params.prompt_text.empty()) {
        safe_print_error_ln("Pipeline error: reference audio was provided without reference text.");
        return false;
    }

    PipelineParams effective_params = params;
    if (ref_audio_buffer && ref_audio_size && *ref_audio_buffer && *ref_audio_size > 0 &&
        effective_params.prompt_audio_cache_key.empty()) {
        effective_params.prompt_audio_cache_key = make_memory_audio_cache_key(*ref_audio_buffer, *ref_audio_size);
    }

    if (ref_audio_buffer && ref_audio_size && *ref_audio_buffer && *ref_audio_size > 0) {
        bool use_cached_prompt = false;
        if (effective_params.permanent_prompt) {
            std::lock_guard<std::mutex> cache_lock(permanent_prompt_cache_mutex_);
            use_cached_prompt = cache_matches(permanent_prompt_cache_, effective_params.prompt_audio_cache_key, effective_params.prompt_text);
        }
        if (use_cached_prompt) {
            safe_print_ln("Using cached permanent reference audio...");
        } else {
            safe_print_ln("Loading reference audio...");
            if (!load_audio_from_memory(*ref_audio_buffer, *ref_audio_size, ref_audio, codec_.sample_rate())) {
                safe_print_error_ln("Pipeline warning: load_audio failed, running without reference audio.");
            }
        }
    }

    int32_t AudioOutFrames = 0;
    if (!this->synthesize_raw(effective_params, ref_audio, audio_out, &AudioOutFrames)) {
        safe_print_error_ln("Pipeline error: synthesis failed.");
        return false;
    }

    if (params.trim_silence && !audio_out.empty()) {
        auto trimmed = audio_trim_trailing_silence(audio_out.data(), audio_out.size(), codec_.sample_rate());
        if (!trimmed.empty()) audio_out = std::move(trimmed);
    }

    if (params.normalize_dynamic && !audio_out.empty()) {
        audio_out = audio_normalize_dynamic(audio_out.data(), audio_out.size(), codec_.sample_rate());
    } else if (params.normalize_output && !audio_out.empty()) {
        float peak = 0.0f;
        for (float s : audio_out) { float a = std::fabs(s); if (a > peak) peak = a; }
        if (peak > 1e-6f) { float scale = 0.95f / peak; for (float & s : audio_out) s *= scale; }
    }

    if (!audio_write_memory_wav(wav_buffer, wav_size, audio_out.data(), audio_out.size(), codec_.sample_rate())) {
        safe_print_error_ln("Pipeline error: audio_write_memory_wav failed");
        return false;
    }

    safe_print_ln("Audio synthesized");
    return true;
}

bool Pipeline::synthesize_raw(const PipelineParams & params, AudioData & ref_audio, std::vector<float>& audio_out, int32_t* audio_out_length) {
    std::lock_guard<std::mutex> lock(synthesize_mutex_);

    if (!initialized_) {
        safe_print_error_ln("Pipeline not initialized.");
        return false;
    }

    model_.clear_kv_cache();

    safe_print_ln("--- Pipeline Synthesize ---");
    safe_print_ln("Text: " + params.text);

    const int32_t num_codebooks = model_.hparams().num_codebooks;

    std::vector<int32_t> ref_codes;
    int32_t T_prompt = 0;
    const bool has_prompt_identity = !params.prompt_audio_cache_key.empty() && !params.prompt_text.empty();
    bool can_use_cache = false;

    {
        std::lock_guard<std::mutex> cache_lock(permanent_prompt_cache_mutex_);
        can_use_cache = params.permanent_prompt &&
            cache_matches(permanent_prompt_cache_, params.prompt_audio_cache_key, params.prompt_text);

        if (!params.permanent_prompt) {
            permanent_prompt_cache_ = PermanentPromptCache{};
        }

        if (can_use_cache) {
            ref_codes = permanent_prompt_cache_.ref_codes;
            T_prompt = permanent_prompt_cache_.t_prompt;
        }
    }

    if (can_use_cache) {
        // Reuse the cached prompt codes below.
    } else if (!ref_audio.samples.empty()) {
        if (!codec_.encode(ref_audio.samples.data(), (int32_t)ref_audio.samples.size(),
                           params.gen.n_threads, ref_codes, T_prompt)) {
            safe_print_error_ln("Pipeline warning: encode failed, running without reference audio.");
            ref_codes.clear();
            T_prompt = 0;
            if (params.permanent_prompt) {
                std::lock_guard<std::mutex> cache_lock(permanent_prompt_cache_mutex_);
                permanent_prompt_cache_ = PermanentPromptCache{};
            }
        } else if (params.permanent_prompt && has_prompt_identity) {
            std::lock_guard<std::mutex> cache_lock(permanent_prompt_cache_mutex_);
            permanent_prompt_cache_.valid = true;
            permanent_prompt_cache_.audio_key = params.prompt_audio_cache_key;
            permanent_prompt_cache_.prompt_text = params.prompt_text;
            permanent_prompt_cache_.ref_codes = ref_codes;
            permanent_prompt_cache_.t_prompt = T_prompt;
            safe_print_ln("Permanent reference audio cached.");
        }
    } else if (params.permanent_prompt) {
        std::lock_guard<std::mutex> cache_lock(permanent_prompt_cache_mutex_);
        permanent_prompt_cache_ = PermanentPromptCache{};
    }

    PromptTensor prompt = build_prompt(
        tokenizer_, params.text, params.prompt_text,
        ref_codes.empty() ? nullptr : ref_codes.data(),
        num_codebooks, T_prompt);

    int32_t max_seq_len = prompt.cols + params.gen.max_new_tokens;
    if (!model_.init_kv_cache(max_seq_len)) {
        safe_print_error_ln("Pipeline error: init_kv_cache failed.");
        return false;
    }

    GenerateResult res = generate(model_, tokenizer_.config(), prompt, params.gen);

    if (res.n_frames == 0) {
        safe_print_error_ln("Pipeline error: generation produced no frames.");
        return false;
    }

    if (!codec_.decode(res.codes.data(), res.n_frames, params.gen.n_threads, audio_out, audio_out_length)) {
        safe_print_error_ln("Pipeline error: decode failed.");
        return false;
    }

    model_.clear_kv_cache();
    return true;
}

}
