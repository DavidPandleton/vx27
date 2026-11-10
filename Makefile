CC       := gcc
CFLAGS   ?= -std=c11 -O3 -DNDEBUG -Wall -Wextra -march=znver3 -mavx2 -mfma -mf16c -mbmi2 -D_WIN32_WINNT=0x0A00
CPPFLAGS += -Isrc
LDFLAGS  ?= -static -static-libgcc -s
ASM      := nasm
ASMFLAGS := -f win64 -O3
BIN      := build/vx27.exe
FORGE    := build/vx_forge.exe
T_ST     := build/test_st.exe
T_DT     := build/test_deltanet.exe
T_DTEX   := build/test_deltanet_exact.exe
T_KV     := build/test_kv.exe
T_ROPE   := build/test_rope.exe
T_GQA    := build/test_gqa.exe
T_CFG    := build/test_cfg.exe
T_FFN    := build/test_ffn.exe
T_HEAD   := build/test_head.exe
T_BLOCK  := build/test_block.exe
VULKAN ?= C:/Users/Public/vulkan
T_QMAT   := build/test_qmat.exe
T_F64    := build/test_forward64.exe
T_GPU    := build/test_gpu.exe
T_VX     := build/test_vxfile.exe
T_GTILED := build/test_gpu_tiled.exe
T_BP3    := build/test_bp3.exe
T_DL     := build/test_dl.exe
T_BATCH  := build/test_batch.exe
T_GFULL  := build/test_gpu_full.exe
T_FORGE  := build/test_forge.exe
T_PERSIST := build/test_persist.exe
T_SFWD   := build/test_sample_fwd.exe
T_S64    := build/test_sample64_fwd.exe
T_REAL   := build/test_real_layer.exe
T_RSHARD := build/test_real_shard.exe
T_FULL   := build/test_full_model.exe
T_FUSED0 := build/test_fused0.exe
T_QFWD64 := build/test_qfwd64.exe
T_HYBRID := build/test_qfwd_hybrid.exe
T_PMAT   := build/test_pmat_reg.exe
T_PMATB  := build/test_pmat_batch.exe
T_QGEN   := build/test_qgen.exe
T_Q8GPU  := build/test_gpu_q8.exe
T_BP3GPU := build/test_gpu_bp3.exe
T_TOK    := build/test_tok.exe
T_SAMPLER := build/test_sampler.exe
T_RUNSMOKE := build/test_run_smoke.exe
T_MTP := build/test_mtp.exe
T_MTPB := build/test_mtp_batch.exe
T_MTPKV := build/test_mtp_kv.exe
T_BD    := build/test_breakdown.exe
T_REFUSAL := build/test_refusal.exe
GEN_REAL := build/gen_real_gate.exe

SRC := src/vx27.c src/vx27_run.c src/vx27_qfwd.c src/vx27_deltanet.c src/vx27_deltanet_avx2.c src/vx27_mtp.c src/vx27_pmat.c src/vx27_gpu_reg.c src/vx27_rope.c src/vx27_ffn.c src/vx27_vxfile.c src/vx27_mmap.c src/vx27_st.c src/vx27_bp3.c src/tokenizer.c src/nfc.c src/vx27_tokbin.c src/gguf.c src/sampler.c src/threadpool.c src/quant.c src/quant_avx2.c
OBJ := $(SRC:src/%.c=build/%.o) build/quant_avx2_asm.o
FORGE_SRC := src/vx_forge.c src/vx27_st.c src/vx27_vxfile.c src/vx27_bp3.c src/quant.c src/quant_avx2.c
FORGE_OBJ := $(FORGE_SRC:src/%.c=build/%.o) build/quant_avx2_asm.o

.PHONY: all clean info test

all: $(BIN) $(FORGE)

$(BIN): $(OBJ) build/bp3_matvec_tiled.spv
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) $(OBJ) -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

$(FORGE): $(FORGE_OBJ)
	$(CC) $(FORGE_OBJ) -o $@ $(LDFLAGS)

build/%.o: src/%.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

build/quant_avx2_asm.o: src/quant_avx2.asm
	$(ASM) $(ASMFLAGS) $< -o $@

build/vx27_gpu.o build/vx27_gpu_full.o build/vx27_gpu_dl.o build/vx27_gpu_reg.o: CPPFLAGS += -I"$(VULKAN)/include"

info: $(BIN) $(FORGE)
	./$(BIN) info
	./$(FORGE) plan

test: $(BIN) $(FORGE) $(T_ST) $(T_DT) $(T_DTEX) $(T_KV) $(T_ROPE) $(T_GQA) $(T_CFG) $(T_FFN) $(T_HEAD) $(T_BLOCK) $(T_QMAT) $(T_F64) $(T_GPU) $(T_VX) $(T_GTILED) $(T_BP3) $(T_DL) $(T_BATCH) $(T_GFULL) $(T_FORGE) $(T_PERSIST) $(T_SFWD) $(T_S64) $(T_REAL) $(T_RSHARD) $(T_FULL) $(T_FUSED0) $(T_QFWD64) $(T_PMAT) $(T_PMATB) $(T_QGEN) $(T_Q8GPU) $(T_BP3GPU) $(T_HYBRID) $(T_TOK) $(T_SAMPLER) $(T_RUNSMOKE) $(T_MTP) $(T_MTPB) $(T_MTPKV) $(T_BD) $(T_REFUSAL)
	./$(T_ST)
	./$(T_DT)
	./$(T_DTEX)
	./$(T_KV)
	./$(T_ROPE)
	./$(T_GQA)
	./$(T_CFG)
	./$(T_FFN)
	./$(T_HEAD)
	./$(T_BLOCK)
	./$(T_QMAT)
	./$(T_F64)
	./$(T_GPU)
	./$(T_VX)
	./$(T_GTILED)
	./$(T_BP3)
	./$(T_DL)
	./$(T_BATCH)
	./$(T_GFULL)
	./$(T_FORGE)
	./$(T_PERSIST)
	./$(T_SFWD)
	./$(T_S64)
	./$(T_REAL)
	./$(T_RSHARD)
	./$(T_FULL)
	./$(T_FUSED0)
	./$(T_QFWD64)
	./$(T_PMAT)
	./$(T_PMATB)
	./$(T_QGEN)
	./$(T_Q8GPU)
	./$(T_BP3GPU)
	./$(T_HYBRID)
	./$(T_TOK)
	./$(T_SAMPLER)
	./$(T_RUNSMOKE)
	./$(T_MTP)
	./$(T_MTPB)
	./$(T_MTPKV)
	./$(T_BD)
	./$(T_REFUSAL)
	./$(BIN) info
	./$(FORGE) plan --out build/manifest.json

sample:
	python3 tools/make_sample.py
	NLAYERS=64 python3 tools/make_sample64.py

sample-real: $(GEN_REAL)
	./$(GEN_REAL)

$(GEN_REAL): tools/gen_real_gate.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(T_ST): tests/test_st.c build/vx27_st.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(T_DT): tests/test_deltanet.c build/vx27_deltanet.o build/vx27_deltanet_avx2.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(T_DTEX): tests/test_deltanet_exact.c build/vx27_deltanet.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(T_KV): tests/test_kv.c build/vx27_kv.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(T_ROPE): tests/test_rope.c build/vx27_rope.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(T_GQA): tests/test_gqa.c build/vx27_gqa.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(T_CFG): tests/test_cfg.c build/vx27_cfg.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(T_FFN): tests/test_ffn.c build/vx27_ffn.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(T_HEAD): tests/test_head.c build/vx27_head.o build/vx27_ffn.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(T_BLOCK): tests/test_block.c build/vx27_block.o build/vx27_rope.o build/vx27_gqa.o build/vx27_ffn.o build/vx27_deltanet.o build/vx27_deltanet_avx2.o build/threadpool.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(T_QMAT): tests/test_qmat.c build/vx27_qmat.o build/vx27_weights.o build/gguf.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(T_F64): tests/test_forward64.c build/vx27_forward64.o build/vx27_block.o build/vx27_rope.o build/vx27_gqa.o build/vx27_ffn.o build/vx27_deltanet.o build/vx27_deltanet_avx2.o build/threadpool.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

build/q4k_matvec.spv: gpu/q4k_matvec.comp
	"$(VULKAN)/bin/glslang.exe" -V -o $@ $<

build/q4k_matvec_tiled.spv: gpu/q4k_matvec_tiled.comp
	"$(VULKAN)/bin/glslang.exe" -V -o $@ $<

build/q4k_matvec_tiled8.spv: gpu/q4k_matvec_tiled8.comp
	"$(VULKAN)/bin/glslang.exe" -V -o $@ $<

build/q8_matvec_tiled.spv: gpu/q8_matvec_tiled.comp
	"$(VULKAN)/bin/glslang.exe" -V -o $@ $<

$(T_GPU): tests/test_gpu.c build/vx27_gpu.o build/gguf.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o build/q4k_matvec.spv
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) tests/test_gpu.c build/vx27_gpu.o build/gguf.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

$(T_VX): tests/test_vxfile.c build/vx27_vxfile.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(T_GTILED): tests/test_gpu_tiled.c build/vx27_gpu.o build/gguf.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o build/q4k_matvec.spv build/q4k_matvec_tiled.spv
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) tests/test_gpu_tiled.c build/vx27_gpu.o build/gguf.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

$(T_BP3): tests/test_bp3.c build/vx27_bp3.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(T_DL): tests/test_dl.c build/vx27_gpu_dl.o
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) tests/test_dl.c build/vx27_gpu_dl.o -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

$(T_BATCH): tests/test_batch.c build/vx27_batch.o build/vx27_gpu.o build/vx27_gpu_full.o build/vx27_gpu_persist.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_bp3.o build/threadpool.o build/gguf.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o build/q4k_matvec.spv build/q4k_matvec_tiled.spv build/q4k_matvec_tiled8.spv
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) tests/test_batch.c build/vx27_batch.o build/vx27_gpu.o build/vx27_gpu_full.o build/vx27_gpu_persist.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_bp3.o build/threadpool.o build/gguf.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

build/vx27_gpu_full.o: src/vx27_gpu_full.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -I"$(VULKAN)/include" -c $< -o $@

$(T_GFULL): tests/test_gpu_full.c build/vx27_gpu_full.o build/vx27_gpu.o build/gguf.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o build/q4k_matvec.spv build/q4k_matvec_tiled.spv build/q4k_matvec_tiled8.spv
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) tests/test_gpu_full.c build/vx27_gpu_full.o build/vx27_gpu.o build/gguf.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

build/vx27_gpu_persist.o: src/vx27_gpu_persist.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -I"$(VULKAN)/include" -c $< -o $@

build/vx27_gpu_reg.o: src/vx27_gpu_reg.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -I"$(VULKAN)/include" -c $< -o $@

$(T_FORGE): tests/test_forge.c build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(T_PERSIST): tests/test_persist.c build/vx27_gpu_persist.o build/gguf.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o build/q4k_matvec_tiled.spv
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) tests/test_persist.c build/vx27_gpu_persist.o build/gguf.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

$(T_SFWD): tests/test_sample_fwd.c build/vx27_sample_fwd.o build/vx27_block.o build/vx27_rope.o build/vx27_gqa.o build/vx27_ffn.o build/vx27_deltanet.o build/vx27_deltanet_avx2.o build/threadpool.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(T_S64): tests/test_sample64_fwd.c build/vx27_block.o build/vx27_rope.o build/vx27_gqa.o build/vx27_ffn.o build/vx27_deltanet.o build/vx27_deltanet_avx2.o build/threadpool.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(T_REAL): tests/test_real_layer.c build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_batch.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_bp3.o build/vx27_gpu.o build/vx27_gpu_full.o build/vx27_gpu_persist.o build/threadpool.o build/gguf.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o build/q4k_matvec.spv build/q4k_matvec_tiled.spv build/q4k_matvec_tiled8.spv
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) tests/test_real_layer.c build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_batch.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_bp3.o build/vx27_gpu.o build/vx27_gpu_full.o build/vx27_gpu_persist.o build/threadpool.o build/gguf.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

$(T_RSHARD): tests/test_real_shard.c build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/gguf.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o build/threadpool.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(T_FULL): tests/test_full_model.c build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(T_FUSED0): tests/test_fused0.c build/vx27_fused.o build/vx27_ffn.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(T_QFWD64): tests/test_qfwd64.c build/vx27_qfwd.o build/vx27_deltanet.o build/vx27_deltanet_avx2.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_rope.o build/vx27_ffn.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o build/bp3_matvec_tiled.spv
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) tests/test_qfwd64.c build/vx27_qfwd.o build/vx27_deltanet.o build/vx27_deltanet_avx2.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_rope.o build/vx27_ffn.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

$(T_TOK): tests/test_tok.c build/tokenizer.o build/nfc.o build/gguf.o build/vx27_tokbin.o build/tok_vocab.bin
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_tok.c build/tokenizer.o build/nfc.o build/gguf.o build/vx27_tokbin.o -o $@ $(LDFLAGS)

build/tok_vocab.bin build/tok_types.bin build/tok_merges.bin build/tok_golden.txt: build/tok.json tools/make_tokbin.py
	python3 tools/make_tokbin.py

$(T_SAMPLER): tests/test_sampler.c build/sampler.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(T_RUNSMOKE): tests/test_run_smoke.c build/vx27_run.o build/vx27_qfwd.o build/vx27_deltanet.o build/vx27_deltanet_avx2.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_rope.o build/vx27_ffn.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/tokenizer.o build/nfc.o build/gguf.o build/vx27_tokbin.o build/sampler.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o build/bp3_matvec_tiled.spv
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) tests/test_run_smoke.c build/vx27_run.o build/vx27_qfwd.o build/vx27_deltanet.o build/vx27_deltanet_avx2.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_rope.o build/vx27_ffn.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/tokenizer.o build/nfc.o build/gguf.o build/vx27_tokbin.o build/sampler.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

$(T_MTP): tests/test_mtp.c build/vx27_run.o build/vx27_mtp.o build/vx27_qfwd.o build/vx27_deltanet.o build/vx27_deltanet_avx2.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_rope.o build/vx27_ffn.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/tokenizer.o build/nfc.o build/gguf.o build/vx27_tokbin.o build/sampler.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o build/bp3_matvec_tiled.spv
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) tests/test_mtp.c build/vx27_run.o build/vx27_mtp.o build/vx27_qfwd.o build/vx27_deltanet.o build/vx27_deltanet_avx2.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_rope.o build/vx27_ffn.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/tokenizer.o build/nfc.o build/gguf.o build/vx27_tokbin.o build/sampler.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

$(T_MTPB): tests/test_mtp_batch.c build/vx27_run.o build/vx27_mtp.o build/vx27_qfwd.o build/vx27_deltanet.o build/vx27_deltanet_avx2.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_rope.o build/vx27_ffn.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/tokenizer.o build/nfc.o build/gguf.o build/vx27_tokbin.o build/sampler.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o build/bp3_matvec_tiled.spv
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) tests/test_mtp_batch.c build/vx27_run.o build/vx27_mtp.o build/vx27_qfwd.o build/vx27_deltanet.o build/vx27_deltanet_avx2.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_rope.o build/vx27_ffn.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/tokenizer.o build/nfc.o build/gguf.o build/vx27_tokbin.o build/sampler.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

$(T_MTPKV): tests/test_mtp_kv.c build/vx27_run.o build/vx27_mtp.o build/vx27_qfwd.o build/vx27_deltanet.o build/vx27_deltanet_avx2.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_rope.o build/vx27_ffn.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/tokenizer.o build/nfc.o build/gguf.o build/vx27_tokbin.o build/sampler.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o build/bp3_matvec_tiled.spv
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) tests/test_mtp_kv.c build/vx27_run.o build/vx27_mtp.o build/vx27_qfwd.o build/vx27_deltanet.o build/vx27_deltanet_avx2.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_rope.o build/vx27_ffn.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/tokenizer.o build/nfc.o build/gguf.o build/vx27_tokbin.o build/sampler.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

$(T_BD): tests/test_breakdown.c build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_bp3.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o build/bp3_matvec_tiled.spv
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) tests/test_breakdown.c build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_bp3.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

$(T_REFUSAL): tests/test_refusal.c build/vx27_run.o build/vx27_qfwd.o build/vx27_deltanet.o build/vx27_deltanet_avx2.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_rope.o build/vx27_ffn.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/tokenizer.o build/nfc.o build/gguf.o build/vx27_tokbin.o build/sampler.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o build/bp3_matvec_tiled.spv
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) tests/test_refusal.c build/vx27_run.o build/vx27_qfwd.o build/vx27_deltanet.o build/vx27_deltanet_avx2.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_rope.o build/vx27_ffn.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/tokenizer.o build/nfc.o build/gguf.o build/vx27_tokbin.o build/sampler.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

$(T_QGEN): tests/test_qgen.c build/vx27_qfwd.o build/vx27_deltanet.o build/vx27_deltanet_avx2.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_rope.o build/vx27_ffn.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o build/bp3_matvec_tiled.spv
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) tests/test_qgen.c build/vx27_qfwd.o build/vx27_deltanet.o build/vx27_deltanet_avx2.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_rope.o build/vx27_ffn.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

$(T_PMAT): tests/test_pmat_reg.c build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_bp3.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o build/bp3_matvec_tiled.spv
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) tests/test_pmat_reg.c build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_bp3.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

$(T_PMATB): tests/test_pmat_batch.c build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_bp3.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o build/bp3_matvec_tiled.spv
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) tests/test_pmat_batch.c build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_bp3.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

$(T_Q8GPU): tests/test_gpu_q8.c build/vx27_gpu_full.o build/vx27_gpu.o build/vx27_batch.o build/vx27_gpu_persist.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/threadpool.o build/gguf.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o build/q8_matvec_tiled.spv
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) tests/test_gpu_q8.c build/vx27_gpu_full.o build/vx27_gpu.o build/vx27_batch.o build/vx27_gpu_persist.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/threadpool.o build/gguf.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

$(T_HYBRID): tests/test_qfwd_hybrid.c build/vx27_qfwd.o build/vx27_deltanet.o build/vx27_deltanet_avx2.o build/vx27_pmat.o build/vx27_rope.o build/vx27_ffn.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/vx27_gpu_reg.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o build/bp3_matvec_tiled.spv
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) tests/test_qfwd_hybrid.c build/vx27_qfwd.o build/vx27_deltanet.o build/vx27_deltanet_avx2.o build/vx27_pmat.o build/vx27_rope.o build/vx27_ffn.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/vx27_gpu_reg.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

build/bp3_matvec_tiled.spv: gpu/bp3_matvec_tiled.comp
	"$(VULKAN)/bin/glslang.exe" -V -o $@ $<

build/vx27_gpu_bp3.o: src/vx27_gpu_bp3.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -I"$(VULKAN)/include" -c $< -o $@

$(T_BP3GPU): tests/test_gpu_bp3.c build/vx27_gpu_bp3.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o build/bp3_matvec_tiled.spv
	$(CC) -Isrc -I"$(VULKAN)/include" $(CFLAGS) tests/test_gpu_bp3.c build/vx27_gpu_bp3.o build/vx27_pmat.o build/vx27_gpu_reg.o build/vx27_vxfile.o build/vx27_mmap.o build/vx27_st.o build/vx27_bp3.o build/threadpool.o build/quant.o build/quant_avx2.o build/quant_avx2_asm.o -o $@ -L"$(VULKAN)/lib" -lvulkan -static-libgcc -s

clean:
	rm -f build/*.o $(BIN) $(FORGE) $(T_ST) $(T_DT) $(T_DTEX) $(T_KV) $(T_ROPE) $(T_GQA) $(T_CFG) $(T_FFN) $(T_HEAD) $(T_BLOCK) $(T_QMAT) $(T_F64) $(T_GPU) $(T_VX) $(T_GTILED) $(T_BP3) $(T_DL) $(T_BATCH) $(T_GFULL) $(T_FORGE) $(T_PERSIST) $(T_SFWD) $(T_S64) $(T_REAL) $(T_RSHARD) $(T_FULL) $(T_FUSED0) $(T_QFWD64) $(T_PMAT) $(T_PMATB) $(T_QGEN) $(T_Q8GPU) $(T_BP3GPU) $(T_HYBRID) $(T_TOK) $(T_SAMPLER) $(T_RUNSMOKE) $(T_MTP) $(T_MTPB) $(T_MTPKV) $(T_BD) $(T_REFUSAL) $(GEN_REAL) build/manifest.json build/test_st*.safetensors build/test_cfg_*.json build/q4k_matvec.spv build/q4k_matvec_tiled.spv build/q4k_matvec_tiled8.spv build/q8_matvec_tiled.spv build/bp3_matvec_tiled.spv build/test_*.vx27 build/sample_*.vx27 build/real_*.vx27 build/shard0_*.vx27 build/sample_real5120.safetensors build/vx27_core.vx27 build/vx27_head.vx27
