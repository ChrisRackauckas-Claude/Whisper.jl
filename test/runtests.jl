using Whisper
using Whisper.LibWhisper
using Test
using DataDeps

using FileIO
using LibSndFile
using SampledSignals
using StringDistances

Whisper.log_level!(:error)

register(DataDep(
    "WhisperSamples",
    "Whisper sample test files",
    [
        "https://upload.wikimedia.org/wikipedia/commons/2/22/George_W._Bush%27s_weekly_radio_address_%28November_1%2C_2008%29.oga", #gb0
        "https://upload.wikimedia.org/wikipedia/commons/1/1f/George_W_Bush_Columbia_FINAL.ogg",                                     #gb1
        "https://upload.wikimedia.org/wikipedia/en/d/d4/En.henryfphillips.ogg",                                                     #hp0
        "https://cdn.openai.com/whisper/draft-20220913a/micro-machines.wav",                                                        #mm1
        "https://upload.wikimedia.org/wikipedia/commons/c/c1/La_contaminacion_del_agua.ogg"                                         #es1
    ],
    [
        "b844a36b9b0c0d777c64f1d62356bf0b6cad6a0f753627f5a7e7abd17c843f0c",
        "97a6384767e2fc3fb27c7593831aa19115d909fcdb85a1e389359ecc4b92a1e8",
        "753014d9f365a3d49989aecb3d2416f2aa6644909fc2d6289e5b986ee1324472",
        "37de21902b32aa2fc147ccbfdcc0566cc7061fffb2c0b10874f05147c0b9de0f",
        "43ee99686d75fd2976128450cec95a621a70a99b4dbd1c224fb9b35c6549daae",
    ],
    post_fetch_method = [
        file->mv(file, "gb0.oga"),
        file->mv(file, "gb1.ogg"),
        file->mv(file, "hp0.ogg"),
        file->mv(file, "mm1.wav"),
        file->mv(file, "es1.ogg")
    ]
))

# Load an audio file and convert it to what Whisper expects: 16 kHz, mono, Float32.
function load_audio(path)
    s = load(path)
    sout = SampleBuf(Float32, 16000, round(Int, length(s) * (16000 / samplerate(s))), nchannels(s))
    write(SampleBufSink(sout), SampleBufSource(s))
    if nchannels(sout) == 1
        return vec(sout.data)
    else
        return vec(sum(sout.data, dims = 2)) ./ nchannels(sout)
    end
end

similarity(a, b) = compare(a, b, Levenshtein())
clean(s) = strip(replace(s, "[BLANK_AUDIO]" => "", "[ Silence ]" => ""))

# Bundled 11 s clip from whisper.cpp's samples (JFK, public domain). Lets most of the
# suite run without network access to the Wikimedia samples.
const JFK_WAV = joinpath(@__DIR__, "jfk.wav")
const JFK_TEXT = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
const jfk = load_audio(JFK_WAV)

@testset "Whisper.jl" begin

@testset "library" begin
    @test Whisper.version() >= v"1.9"
    @test "large-v3-turbo" in available_models()
    @test "tiny.en" in available_models()
end

@testset "model downloads" begin
    @test isdir(datadep"whisper-ggml-tiny.en")
    @test isfile(datadep"whisper-ggml-tiny.en/ggml-tiny.en.bin")
    @test isfile(Whisper.model_path("tiny.en"))
    @test Whisper.model_path(JFK_WAV) == JFK_WAV          # existing files pass through
    @test_throws ArgumentError Whisper.model_path("no-such-model")
end

@testset "LibWhisper: whisper_full_params accessors" begin
    # whisper_full_params is exposed as opaque bytes plus generated pointer
    # accessors; make sure setting through them round-trips.
    p = Ref(whisper_full_default_params(WHISPER_SAMPLING_GREEDY))
    GC.@preserve p begin
        ptr = Base.unsafe_convert(Ptr{whisper_full_params}, p)
        ptr.n_threads = Cint(3)
        ptr.translate = true
        ptr.beam_search.beam_size = Cint(7)
        @test unsafe_load(ptr.n_threads) == 3
        @test unsafe_load(ptr.translate) == true
        @test unsafe_load(ptr.beam_search.beam_size) == 7
    end
    @test p[].n_threads == 3
    @test p[].strategy == WHISPER_SAMPLING_GREEDY
    pb = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH)
    @test pb.strategy == WHISPER_SAMPLING_BEAM_SEARCH
end

@testset "WhisperContext" begin
    ctx = WhisperContext("tiny.en")
    @test isopen(ctx)
    @test occursin("tiny.en", sprint(show, ctx))
    @test !Whisper.is_multilingual(ctx)

    r1 = transcribe(ctx, jfk)
    r2 = transcribe(ctx, jfk)                       # context is reusable
    @test r1 == r2
    @test similarity(clean(r1), JFK_TEXT) > 0.9

    segs = segments(ctx)
    @test !isempty(segs)
    @test all(s -> s.t0 <= s.t1, segs)
    @test first(segs).t0 >= 0
    @test similarity(clean(join(s.text for s in segs)), JFK_TEXT) > 0.9

    @test transcribe(ctx, Float32[]) == ""

    close(ctx)
    @test !isopen(ctx)
    @test occursin("closed", sprint(show, ctx))
    @test_throws ArgumentError transcribe(ctx, jfk)
    @test_throws ArgumentError segments(ctx)
    close(ctx)                                       # idempotent
end

@testset "transcribe options" begin
    ctx = WhisperContext("tiny.en"; use_gpu = false)
    @test similarity(clean(transcribe(ctx, jfk; sampling = :beam, beam_size = 3)), JFK_TEXT) > 0.9
    @test similarity(clean(transcribe(ctx, jfk; n_threads = 2)), JFK_TEXT) > 0.9
    @test similarity(clean(transcribe(ctx, jfk; initial_prompt = "JFK inaugural address.")), JFK_TEXT) > 0.85
    @test similarity(clean(transcribe(ctx, jfk; no_timestamps = true)), JFK_TEXT) > 0.9
    @test similarity(clean(transcribe(ctx, jfk; single_segment = true)), JFK_TEXT) > 0.9
    @test similarity(clean(transcribe(ctx, jfk; temperature = 0.2)), JFK_TEXT) > 0.8
    # first ~3 s only
    short = transcribe(ctx, jfk; duration_ms = 3000)
    @test length(short) < length(JFK_TEXT)
    # integer input is accepted and converted
    @test transcribe(ctx, round.(Int16, jfk .* 0)) isa String
    @test_throws ArgumentError transcribe(ctx, jfk; sampling = :nope)
    close(ctx)
end

@testset "one-shot transcribe and GPU flag" begin
    # use_gpu=true must be harmless on a CPU-only build (falls back)
    @test similarity(clean(transcribe("tiny.en", jfk; use_gpu = true)), JFK_TEXT) > 0.9
    @test similarity(clean(transcribe("tiny.en", jfk; use_gpu = false)), JFK_TEXT) > 0.9
    @test_throws ArgumentError transcribe("no-such-model", jfk)
end

@testset "log level" begin
    @test Whisper.log_level!(:none) === nothing
    @test Whisper.log_level!(:error) === nothing
    @test_throws ArgumentError Whisper.log_level!(:loud)
end

@testset "multilingual" begin
    ctx = WhisperContext("base")
    @test Whisper.is_multilingual(ctx)
    r = transcribe(ctx, jfk; language = "auto")
    @test similarity(clean(r), JFK_TEXT) > 0.9
    @test Whisper.detected_language(ctx) == "en"
    es = load_audio(DataDeps.resolve("WhisperSamples/es1.ogg", "__FILE__"))
    transcribe(ctx, es; language = "auto", duration_ms = 20_000)
    @test Whisper.detected_language(ctx) == "es"
    close(ctx)
end

function transcribe_test(ctx, audio_file, txt_file; accuracy = 0.99)
    audio = load_audio(DataDeps.resolve("WhisperSamples/$audio_file", "__FILE__"))
    result = clean(transcribe(ctx, audio))
    expected = readlines(joinpath(@__DIR__, txt_file))[1]
    sim = similarity(expected, result)
    sim > accuracy || @info "similarity $sim below $accuracy" audio_file result
    @test sim > accuracy
end

@testset "Transcription with base.en" begin
    ctx = WhisperContext("base.en")
    transcribe_test(ctx, "gb0.oga", "gb0.txt")
    transcribe_test(ctx, "gb1.ogg", "gb1.txt")
    transcribe_test(ctx, "hp0.ogg", "hp0.txt")
    transcribe_test(ctx, "mm1.wav", "mm1.txt", accuracy = 0.90)
    close(ctx)
end

@testset "Transcription with tiny.en" begin
    ctx = WhisperContext("tiny.en")
    transcribe_test(ctx, "gb0.oga", "gb0.txt", accuracy = 0.90)
    transcribe_test(ctx, "gb1.ogg", "gb1.txt", accuracy = 0.90)
    transcribe_test(ctx, "hp0.ogg", "hp0.txt", accuracy = 0.90)
    close(ctx)
end

end # testset
