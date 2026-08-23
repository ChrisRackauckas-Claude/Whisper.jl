module Whisper

using DataDeps

include("LibWhisper.jl")
using .LibWhisper

include("models.jl")

export WhisperContext, transcribe, segments, available_models

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# whisper.cpp (and ggml underneath it) log through a C callback. The default
# callback prints everything, including several dozen lines of model metadata on
# every load. We install our own and filter by level.
const LOG_LEVELS = (none = 5, error = 4, warn = 3, info = 2, debug = 1)
const _log_threshold = Ref{Cint}(LOG_LEVELS.warn)
const _log_callback = Ref{Ptr{Cvoid}}(C_NULL)

function _log_cb(level::Cint, text::Cstring, ::Ptr{Cvoid})::Cvoid
    # GGML_LOG_LEVEL_CONT (5) continues the previous message; treat it as info.
    lvl = level == 5 ? Cint(2) : level
    if lvl >= _log_threshold[]
        print(stderr, unsafe_string(text))
    end
    return nothing
end

"""
    Whisper.log_level!(level::Symbol)

Set how much of whisper.cpp's own logging reaches `stderr`. One of `:debug`,
`:info`, `:warn` (default), `:error`, `:none`. `:info` shows model and backend
details on load, including whether the GPU was used.
"""
function log_level!(level::Symbol)
    haskey(LOG_LEVELS, level) ||
        throw(ArgumentError("log level must be one of $(keys(LOG_LEVELS)), got :$level"))
    _log_threshold[] = LOG_LEVELS[level]
    return nothing
end

function __init__()
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    register_datadeps()
    _log_callback[] = @cfunction(_log_cb, Cvoid, (Cint, Cstring, Ptr{Cvoid}))
    whisper_log_set(_log_callback[], C_NULL)
    return nothing
end

"""
    Whisper.version() -> VersionNumber

Version of the underlying whisper.cpp library.
"""
version() = VersionNumber(unsafe_string(whisper_version()))

# ---------------------------------------------------------------------------
# Context (a loaded model)
# ---------------------------------------------------------------------------

# Rebuild an immutable C struct with some fields replaced.
function _with(x::T; kw...) where {T}
    vals = ntuple(i -> (f = fieldname(T, i); haskey(kw, f) ? kw[f] : getfield(x, f)), fieldcount(T))
    return T(vals...)
end

"""
    WhisperContext(model; use_gpu=true, gpu_device=0, flash_attn=false)

Load a Whisper model. `model` is a name from [`available_models`](@ref) (downloaded on
first use) or a path to a ggml `.bin` file. The context holds the model in memory so
it can be reused across many [`transcribe`](@ref) calls; it is freed by the garbage
collector, or explicitly with `close`.

- `use_gpu`: run on the GPU when the underlying whisper.cpp build has a GPU backend
  (CUDA, Metal, ...). Silently falls back to the CPU otherwise, so this is safe to
  leave on. Call `Whisper.log_level!(:info)` before loading to see which backend is
  used.
- `gpu_device`: which GPU to use when several are present.
- `flash_attn`: use flash attention (GPU backends only).

A context is not thread-safe: do not call `transcribe` on the same context from
several threads at once.
"""
mutable struct WhisperContext
    ptr::Ptr{whisper_context}
    model::String

    function WhisperContext(model::AbstractString;
                            use_gpu::Bool = true, gpu_device::Integer = 0,
                            flash_attn::Bool = false)
        path = model_path(model)
        cparams = _with(whisper_context_default_params();
                        use_gpu = use_gpu, gpu_device = Cint(gpu_device),
                        flash_attn = flash_attn)
        ptr = whisper_init_from_file_with_params(path, cparams)
        ptr == C_NULL && error("whisper.cpp failed to load model \"$model\" from $path")
        ctx = new(ptr, String(model))
        finalizer(close, ctx)
        return ctx
    end
end

function Base.close(ctx::WhisperContext)
    if ctx.ptr != C_NULL
        whisper_free(ctx.ptr)
        ctx.ptr = C_NULL
    end
    return nothing
end

Base.isopen(ctx::WhisperContext) = ctx.ptr != C_NULL

function _check(ctx::WhisperContext)
    isopen(ctx) || throw(ArgumentError("WhisperContext has been closed"))
    return ctx.ptr
end

Base.show(io::IO, ctx::WhisperContext) =
    print(io, "WhisperContext(\"", ctx.model, "\")", isopen(ctx) ? "" : " (closed)")

"""
    is_multilingual(ctx::WhisperContext) -> Bool

Whether the loaded model supports languages other than English.
"""
is_multilingual(ctx::WhisperContext) = whisper_is_multilingual(_check(ctx)) != 0

# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

"""
    transcribe(ctx::WhisperContext, audio; kwargs...) -> String
    transcribe(model::AbstractString, audio; use_gpu=true, kwargs...) -> String

Run speech recognition on `audio`, a vector of samples at **16 kHz, mono**, with
values in `[-1, 1]` (it is converted to `Vector{Float32}`). Returns the transcript
as a single string; use [`segments`](@ref) afterwards for per-segment timestamps.

The second form loads `model` (see [`available_models`](@ref)), transcribes, and
frees it again; when transcribing more than once, create a [`WhisperContext`](@ref)
and reuse it, since loading the model dominates for short audio.

# Keyword arguments
- `language = "en"`: ISO 639-1 code of the spoken language, or `"auto"` to detect it.
  Only multilingual models (those without the `.en` suffix) understand anything but
  English.
- `translate = false`: translate the transcript to English (multilingual models).
- `n_threads = min(4, Sys.CPU_THREADS)`: CPU threads for the decoder.
- `sampling = :greedy`: `:greedy` or `:beam` (beam search; slower, usually more accurate).
- `best_of = 5`: candidates kept for greedy sampling with temperature fallback.
- `beam_size = 5`: beam width for `sampling = :beam`.
- `temperature = 0.0`: initial sampling temperature.
- `initial_prompt = nothing`: text that primes the decoder (vocabulary, style, names).
- `no_timestamps = false`: skip timestamp tokens (slightly faster). The times reported
  by `segments` are then meaningless; only the text is.
- `offset_ms = 0`, `duration_ms = 0`: transcribe only part of the audio (0 = all).
- `max_len = 0`: maximum characters per segment (0 = no limit).
- `single_segment = false`: force the output into one segment (for short clips).
"""
function transcribe(ctx::WhisperContext, audio::AbstractVector{<:Real};
                    language::Union{AbstractString,Nothing} = "en",
                    translate::Bool = false,
                    n_threads::Integer = min(4, Sys.CPU_THREADS),
                    sampling::Symbol = :greedy,
                    best_of::Integer = 5,
                    beam_size::Integer = 5,
                    temperature::Real = 0.0,
                    initial_prompt::Union{AbstractString,Nothing} = nothing,
                    no_timestamps::Bool = false,
                    offset_ms::Integer = 0,
                    duration_ms::Integer = 0,
                    max_len::Integer = 0,
                    single_segment::Bool = false)
    cptr = _check(ctx)
    sampling in (:greedy, :beam) ||
        throw(ArgumentError("sampling must be :greedy or :beam, got :$sampling"))
    strategy = sampling == :beam ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY

    samples = convert(Vector{Float32}, audio)
    isempty(samples) && return ""

    # Strings pointed to from the params must outlive whisper_full.
    lang = language === nothing ? nothing : String(language)
    prompt = initial_prompt === nothing ? nothing : String(initial_prompt)
    if lang !== nothing && lang != "en" && lang != "auto" && !is_multilingual(ctx)
        @warn "model \"$(ctx.model)\" is English-only; language=\"$lang\" will be ignored" maxlog = 1
    end

    # whisper_full_params has nested anonymous structs, so the binding exposes it
    # as opaque bytes with generated pointer accessors; fill it through a Ref.
    params = Ref(whisper_full_default_params(strategy))
    ret = GC.@preserve params samples lang prompt begin
        p = Base.unsafe_convert(Ptr{whisper_full_params}, params)
        p.n_threads = Cint(n_threads)
        p.translate = translate
        p.language = lang === nothing ? Ptr{Cchar}(C_NULL) : Ptr{Cchar}(pointer(lang))
        p.detect_language = false
        p.no_timestamps = no_timestamps
        p.offset_ms = Cint(offset_ms)
        p.duration_ms = Cint(duration_ms)
        p.max_len = Cint(max_len)
        p.single_segment = single_segment
        p.temperature = Cfloat(temperature)
        p.greedy.best_of = Cint(best_of)
        p.beam_search.beam_size = Cint(beam_size)
        p.initial_prompt = prompt === nothing ? Ptr{Cchar}(C_NULL) : Ptr{Cchar}(pointer(prompt))
        # whisper.cpp prints progress and live results by default; keep quiet.
        p.print_progress = false
        p.print_realtime = false
        p.print_special = false
        p.print_timestamps = false
        whisper_full(cptr, params[], samples, length(samples))
    end
    ret == 0 || error("whisper_full failed with code $ret")

    n = whisper_full_n_segments(cptr)
    io = IOBuffer()
    for i in 0:(n - 1)
        write(io, unsafe_string(whisper_full_get_segment_text(cptr, i)))
    end
    return String(take!(io))
end

function transcribe(model::AbstractString, audio::AbstractVector{<:Real};
                    use_gpu::Bool = true, kwargs...)
    ctx = WhisperContext(model; use_gpu = use_gpu)
    try
        return transcribe(ctx, audio; kwargs...)
    finally
        close(ctx)
    end
end

"""
    segments(ctx::WhisperContext) -> Vector{@NamedTuple{t0::Float64, t1::Float64, text::String}}

The segments produced by the most recent [`transcribe`](@ref) call on `ctx`, with
start and end times in seconds.
"""
function segments(ctx::WhisperContext)
    cptr = _check(ctx)
    n = whisper_full_n_segments(cptr)
    out = Vector{@NamedTuple{t0::Float64, t1::Float64, text::String}}(undef, n)
    for i in 0:(n - 1)
        # whisper.cpp reports times in centiseconds
        t0 = whisper_full_get_segment_t0(cptr, i) / 100
        t1 = whisper_full_get_segment_t1(cptr, i) / 100
        out[i + 1] = (t0 = t0, t1 = t1, text = unsafe_string(whisper_full_get_segment_text(cptr, i)))
    end
    return out
end

"""
    detected_language(ctx::WhisperContext) -> String

ISO 639-1 code of the language whisper.cpp decided on during the most recent
[`transcribe`](@ref) call (meaningful with `language = "auto"`).
"""
detected_language(ctx::WhisperContext) =
    unsafe_string(whisper_lang_str(whisper_full_lang_id(_check(ctx))))

end # module
