# Jetson Nano deployment

The reported deployment used a Jetson Nano B01 with Ubuntu 18.04, L4T R32.7.6, CUDA 10.2, TensorRT 8.2.1, DeepStream 6.0, FP16, batch 1, 640 × 640 input, MAXN mode, and `jetson_clocks` enabled. The formal ONNX was produced with the DeepStream-Yolo YOLOv8 exporter and opset 12.

## 1. Export and verify ONNX

Use the patched, editable yolov10 environment described in the root README. Pin the exporter source exactly:

```bash
export SMFR_REPO=/absolute/path/to/SMFR-YOLO
export DEEPSTREAM_YOLO=/absolute/path/to/DeepStream-Yolo
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git "$DEEPSTREAM_YOLO"
git -C "$DEEPSTREAM_YOLO" checkout 2894babce8e75c49115dbe0c7b516289ed853565
cd "$SMFR_REPO"
python -m pip install -r deployment/requirements.txt
python scripts/export_onnx.py \
  --checkpoint /absolute/path/to/best.pt \
  --deepstream-yolo-root "$DEEPSTREAM_YOLO"
```

The wrapper invokes `DeepStream-Yolo/utils/export_yoloV8.py` with `-s 640 640 --opset 12 --batch 1` and then runs `onnx.checker`. It rejects an export unless the contract is:

```text
input:  float [1, 3, 640, 640]
output: float [1, 8400, 6]
opset:  12
```

An existing export can be checked independently:

```bash
python scripts/check_onnx_contract.py /absolute/path/to/best.onnx
```

## 2. Build the DeepStream-Yolo custom library on Jetson

Use the same pinned DeepStream-Yolo checkout on the Jetson. The parser supplied in this repository adds defensive finite-value and class-range checks for new public builds:

```bash
cp "$SMFR_REPO/deployment/deepstream/parser/nvdsparsebbox_yolo.cpp" \
  "$DEEPSTREAM_YOLO/nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp"

export CUDA_VER=10.2
make -C "$DEEPSTREAM_YOLO/nvdsinfer_custom_impl_Yolo" clean
make -C "$DEEPSTREAM_YOLO/nvdsinfer_custom_impl_Yolo"

nm -D "$DEEPSTREAM_YOLO/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so" \
  | grep -E 'NvDsInferParseYolo|NvDsInferYoloCudaEngineGet'
```

This build compiles the upstream engine builder, parser, plugins, utilities, and CUDA sources into one library. The two symbols printed by the final command match `parse-bbox-func-name` and `engine-create-func-name` in the supplied inference configurations.

The hardened parser is a defensive public-build variant. The final SMFR-YOLO runtime-stability validation kept the frozen formal parser unchanged; parser hardening was not the mechanism used to correct the Jetson Nano runtime corruption. The compatibility correction is the inference-completion patch described below.

## 3. Build and benchmark the FP16 engine

Set MAXN mode and clocks before measurement:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
/usr/src/tensorrt/bin/trtexec \
  --onnx=/absolute/path/to/best.onnx \
  --saveEngine=/absolute/path/to/best_fp16.engine \
  --fp16 --workspace=1024 \
  --warmUp=1000 --duration=60 --streams=1
```

The formal TensorRT protocol used batch 1, one stream, a 1,000 ms warm-up, and a 60 s measurement. Treat the generated engine as platform-specific; rebuild it on the target Jetson rather than publishing it as a portable artifact.

## 4. Apply the DeepStream 6.0 runtime compatibility patch

On the Jetson Nano / TensorRT 8.2.1 / DeepStream 6.0 stack used in this study, the final SMFR-YOLO deployment required an inference-completion boundary before the same TensorRT execution context could proceed to another outstanding submission.

The compatibility patch is intentionally minimal. It starts from the NVIDIA DeepStream 6.0 `nvdsinfer` source and adds `cudaEventSynchronize(*m_InferCompleteEvent)` immediately after each of the two inference-completion event records. It does not modify model inputs, model outputs, the TensorRT engine, `m_InputConsumedEvent`, or detector post-processing.

```bash
export DS_ROOT=/opt/nvidia/deepstream/deepstream-6.0
export NVDSINFER_SRC="$DS_ROOT/sources/libs/nvdsinfer"
export SMFR_RUNTIME=/absolute/path/to/smfr_nvdsinfer_runtime

rm -rf "$SMFR_RUNTIME"
mkdir -p "$SMFR_RUNTIME/src" "$SMFR_RUNTIME/lib"

cp -a "$NVDSINFER_SRC/." "$SMFR_RUNTIME/src/"

cd "$SMFR_RUNTIME/src"
patch -p1 < "$SMFR_REPO/deployment/deepstream/nvdsinfer/event_host_wait.patch"

test "$(grep -Fc 'cudaEventSynchronize(*m_InferCompleteEvent)' nvdsinfer_context_impl.cpp)" -eq 2

make CUDA_VER=10.2 WITH_OPENCV=0 clean

make CUDA_VER=10.2 WITH_OPENCV=0 \
  CFLAGS='-fPIC -Wno-deprecated-declarations -std=c++14 -I /usr/local/cuda-10.2/include -I /opt/nvidia/deepstream/deepstream-6.0/sources/includes -DNDEBUG' \
  -j2

cp libnvds_infer.so "$SMFR_RUNTIME/lib/libnvds_infer.so"

ldd "$SMFR_RUNTIME/lib/libnvds_infer.so" | grep 'not found' && exit 1 || true
```

Do not overwrite `/opt/nvidia/deepstream/deepstream-6.0/lib/libnvds_infer.so`. The validated deployment used a project-scoped candidate library selected through `LD_LIBRARY_PATH`.

## 5. Configure and run DeepStream

In both `deepstream/infer_interval_0.txt` and `deepstream/infer_interval_2.txt`, replace:

- `<SMFR_ONNX_PATH>` with the absolute ONNX path.
- `<SMFR_ENGINE_PATH>` with the absolute FP16 engine path.
- `<PARSER_LIBRARY_PATH>` with the absolute path to `libnvdsinfer_custom_impl_Yolo.so`.

Set the input URI in `deepstream/app_config.txt`, then run from the configuration directory so the relative label and inference-config paths resolve:

```bash
export DS_ROOT=/opt/nvidia/deepstream/deepstream-6.0

export LD_LIBRARY_PATH="$SMFR_RUNTIME/lib:$DS_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

GST_INFER_PLUGIN=$(
  find "$DS_ROOT" \
    -type f \
    -name 'libnvdsgst_infer.so' \
    -print \
    -quit
)

test -n "$GST_INFER_PLUGIN"

ldd "$GST_INFER_PLUGIN" | grep 'libnvds_infer'

cd "$SMFR_REPO/deployment/deepstream"
"$DS_ROOT/bin/deepstream-app" -c app_config.txt

```

`app_config.txt` selects `infer_interval_0.txt` by default. Change its `config-file` entry to `infer_interval_2.txt` to infer every third frame. Both inference files keep batch 1, FP16 (`network-mode=2`), three classes, confidence 0.25, NMS IoU 0.45, top-k 300, aspect-ratio preservation, and symmetric padding. DeepStream performance reporting is enabled at a five-second interval; use the cumulative FPS from the same input video and exclude one-time engine construction from the timed run.

## 6. Historical results

The SMFR-YOLO deployment row reports 16.5416 TensorRT QPS, 59.788 ms mean latency, 60.453 ms end-to-end engine time, 12.1967 ms enqueue time, 59.275 ms GPU compute time, 15.79 FPS at interval 0, and 42.76 FPS at interval 2. The complete nine-model comparison is in `results/deployment/nano_benchmark.tsv`.

ONNX files, TensorRT engines, checkpoints, and compiled libraries are local build products. The commands above are a reproducible build path; the historical runtime numbers require Jetson Nano hardware to revalidate.The Table 10 throughput and FPS values are historical benchmark measurements and should not be reinterpreted as measurements of the later runtime-compatibility patch itself. The EVENT_HOST_WAIT validation established stable execution on the legacy Nano runtime stack; it is a reliability correction, whereas the reported Table 10 values remain the paper's frozen performance measurements.
