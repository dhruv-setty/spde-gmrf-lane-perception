#!/usr/bin/env bash
# Produces the README figures from the sources in this repository, on a real X display
# (Xvfb). Terminal figures are screenshots (ImageMagick import) of an xterm running the
# named commands; plots are written by docs/capture/plots.py with the Python reference in
# python/spdegmrf; the CARLA frame grid is cut from the bundled clips with ffmpeg. No figure
# is drawn by hand. The log of every terminal run is kept next to its image.
#
#   docs/capture/capture.sh            all captures (needs Xvfb, xterm, ImageMagick)
#   docs/capture/capture.sh step NAME  one terminal step, printed to stdout
#
# Nothing here needs a GPU, a phone or a board. The firmware runs under qemu-arm (user-mode
# emulation of a Cortex-M4 instruction set with semihosting), which gives no cycle counts.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=build/capture; shots=docs/screenshots; mkdir -p "$OUT" "$shots"
M4=(-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16)
CMSIS_NN_TAG=v8.0.0
CMSIS_6_TAG=v6.4.0

run() { echo "\$ $*"; "$@"; local rc=$?; echo "[exit $rc]"; return $rc; }

step() {
  case "$1" in
  pytest)
    python --version; export CC=gcc; echo "CC=gcc"
    run python -m pytest -o addopts= -v -s -p no:cacheprovider
    ;;
  fw_host)
    gcc --version | head -1
    run make -C firmware clean test CC=gcc WERROR=1
    ;;
  fw_m4_size)
    local b="$OUT/m4"; rm -rf "$b"; mkdir -p "$b"
    printf '#include "hmm_filter.h"\n#include "hmm_model_generated.h"\nconst hmm_model_t *const hmm_model_ptr = &hmm_gen_model;\nhmm_state_t hmm_state;\n' > "$b/model.c"
    arm-none-eabi-gcc --version | head -1
    echo "flags: ${M4[*]} -Os -std=c99 -fstack-usage -fcallgraph-info=su"
    for s in firmware/src/hmm_filter.c firmware/src/hmm_fc_s8_ref.c firmware/src/hmm_exp_lut.c "$b/model.c"; do
      arm-none-eabi-gcc "${M4[@]}" -Os -std=c99 -Wall -Wextra -fstack-usage -fcallgraph-info=su -Ifirmware/include -c "$s" -o "$b/$(basename "${s%.c}").o" || echo "[compile failed: $s]"
    done
    (cd "$b" && run arm-none-eabi-size -t hmm_filter.o hmm_fc_s8_ref.o hmm_exp_lut.o model.o)
    echo "\$ cat *.su   (gcc -fstack-usage: function, bytes, qualifier)"
    cat "$b"/*.su | sed "s#^.*/##"
    echo "\$ grep edge *.ci   (gcc -fcallgraph-info: calls made by the filter)"
    grep -h "^edge" "$b"/*.ci | sed -E 's/.*sourcename: "([^"]*)" targetname: "([^"]*)".*/\1 -> \2/'
    echo "# filter, model and state linked alone (entry hmm_step, --gc-sections, libgcc for 64-bit division)"
    (cd "$b" && run arm-none-eabi-gcc "${M4[@]}" -Os -nostartfiles -nostdlib -Wl,--gc-sections -Wl,-e,hmm_step \
      -Wl,--undefined=hmm_init -Wl,--undefined=hmm_mode -Wl,--undefined=hmm_model_ptr -Wl,--undefined=hmm_state \
      -o filter.elf hmm_filter.o hmm_fc_s8_ref.o hmm_exp_lut.o model.o -lgcc)
    (cd "$b" && run arm-none-eabi-size -A filter.elf) | grep -vE '^\.(debug|comment)|^\.ARM\.attributes|^Total|^$'
    (cd "$b" && arm-none-eabi-nm -S --size-sort filter.elf | grep -E ' (T|W) __')
    (cd firmware && export CC_M4="arm-none-eabi-gcc ${M4[*]}" && echo "CC_M4=$CC_M4" && run python tools/footprint.py)
    ;;
  fw_m4_qemu)
    local b="$OUT/qemu"; rm -rf "$b"; mkdir -p "$b"
    local src=(firmware/src/hmm_filter.c firmware/src/hmm_fc_s8_ref.c firmware/src/hmm_exp_lut.c firmware/tests/test_hmm_filter.c)
    local f=("${M4[@]}" -Os -std=c99 -Ifirmware/include -Ifirmware/tests --specs=rdimon.specs)
    arm-none-eabi-gcc --version | head -1
    qemu-arm --version | head -1
    echo "# 1. portable C99 kernel (hmm_fc_s8_ref.c), Cortex-M4F, newlib rdimon semihosting"
    run arm-none-eabi-gcc "${f[@]}" -o "$b/test_ref.elf" "${src[@]}"
    run qemu-arm -cpu cortex-m4 -B 0x40000000 "$b/test_ref.elf"
    echo "# runtime 64-bit division linked into test_ref.elf (nm -S: address, size, type, name)"
    arm-none-eabi-nm -S "$b/test_ref.elf" | grep -E ' __aeabi_ldivmod$| __aeabi_uldivmod$| __gnu_ldivmod_helper$| __divmoddi4$| __udivmoddi4$' || true
    echo "# 2. CMSIS-NN $CMSIS_NN_TAG arm_fully_connected_s8 (HMM_USE_CMSIS_NN), CMSIS_6 $CMSIS_6_TAG core headers"
    [ -d "$OUT/CMSIS-NN" ] || git clone -q --depth 1 -b "$CMSIS_NN_TAG" https://github.com/ARM-software/CMSIS-NN.git "$OUT/CMSIS-NN" 2>/dev/null
    [ -d "$OUT/CMSIS_6" ] || git clone -q --depth 1 -b "$CMSIS_6_TAG" https://github.com/ARM-software/CMSIS_6.git "$OUT/CMSIS_6" 2>/dev/null
    local nn="$OUT/CMSIS-NN"
    local nnsrc=("$nn/Source/FullyConnectedFunctions/arm_fully_connected_s8.c"
                 "$nn/Source/FullyConnectedFunctions/arm_fully_connected_get_buffer_sizes_s8.c"
                 "$nn/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c")
    run arm-none-eabi-gcc "${f[@]}" -DHMM_USE_CMSIS_NN -I"$nn/Include" -I"$OUT/CMSIS_6/CMSIS/Core/Include" \
      -o "$b/test_cmsis.elf" "${src[@]}" firmware/src/hmm_fc_s8_cmsis.c "${nnsrc[@]}"
    run qemu-arm -cpu cortex-m4 -B 0x40000000 "$b/test_cmsis.elf"
    echo "# CMSIS-NN backend objects, Cortex-M4F -Os"
    for s in firmware/src/hmm_fc_s8_cmsis.c "${nnsrc[@]}"; do
      arm-none-eabi-gcc "${M4[@]}" -Os -std=c99 -DHMM_USE_CMSIS_NN -Ifirmware/include -I"$nn/Include" \
        -I"$OUT/CMSIS_6/CMSIS/Core/Include" -c "$s" -o "$b/$(basename "${s%.c}").o"
    done
    (cd "$b" && run arm-none-eabi-size -t hmm_fc_s8_cmsis.o arm_fully_connected_s8.o arm_fully_connected_get_buffer_sizes_s8.o arm_nn_vec_mat_mult_t_s8.o)
    ;;
  mac_table)
    run python python/tools/mac_table.py
    ;;
  nvcc)
    if ! command -v nvcc >/dev/null; then echo "nvcc not installed: CUDA compile not captured"; return 0; fi
    nvcc --version | tail -2
    # The source uses the cuDSS 0.x signature of cudssMatrixCreateCsr; it is compiled against
    # the last 0.7 wheel and against the current wheel. Compile and link only: no GPU here.
    local v d
    for v in 0.7.1.6 1.0.0.35; do
      d="$OUT/cudss-$v"
      pip install -q --no-deps --target "$d" "nvidia-cudss-cu12==$v" >/dev/null 2>&1 || echo "pip install nvidia-cudss-cu12==$v failed"
      echo "# cuDSS $v (PyPI wheel nvidia-cudss-cu12==$v)"
      run nvcc -std=c++17 -O2 -arch=sm_80 -I"$d/nvidia/cu12/include" -c gpu/src/spde_gpu.cu -o "$OUT/spde_gpu_$v.o" 2>&1 \
        | grep -E '^\$|error:|errors detected|^\[exit' | cut -c1-148
      if [ -f "$OUT/spde_gpu_$v.o" ]; then
        run nvcc -arch=sm_80 -o "$OUT/spde_gpu_$v" "$OUT/spde_gpu_$v.o" -L"$d/nvidia/cu12/lib" -l:libcudss.so.1 -lcusparse \
          -Xlinker --allow-shlib-undefined
        (cd "$OUT" && run ls -l "spde_gpu_$v.o" "spde_gpu_$v")
      fi
    done
    echo "Not run: this runner has no GPU."
    ;;
  *) echo "unknown step $1"; return 2 ;;
  esac
}

if [ "${1:-}" = step ]; then step "$2"; exit $?; fi

# Tool versions.
{ python --version; python -c "import numpy, scipy, matplotlib; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'matplotlib', matplotlib.__version__)"
  gcc --version | head -1; arm-none-eabi-gcc --version | head -1; qemu-arm --version | head -1
  convert -version | head -1; xterm -version
  command -v nvcc >/dev/null && nvcc --version | tail -2; } > "$shots/versions.log" 2>&1

# 1. Plots from the Python reference.
python docs/capture/plots.py "$shots" || echo "plots.py failed" >> "$shots/plots.log"
python python/tools/fill_in_figure.py "$shots/plot_fill_in_regenerated.png" | tee -a "$shots/plots.log"

# 2. Still frames from the four CARLA clips: frames 0, 100, 200 and 299 of each (t = n / 20 s),
#    scaled to the 320 x 192 capture resolution, one row per clip. ffmpeg is the static build
#    shipped with the imageio-ffmpeg wheel; labels are added with ImageMagick.
FFMPEG=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
"$FFMPEG" -version | head -1 >> "$shots/versions.log"
rows=()
: > "$shots/carla_frame_grid.log"
for clip in carla_town10hd_urban_clear_ep04 carla_town10hd_urban_clear_ep09 carla_town10hd_urban_rain_sunset_ep19 carla_town04_highway_wet_sunset_ep18; do
  "$FFMPEG" -hide_banner -i "assets/videos/$clip.mp4" 2>&1 | grep -E "Duration|Stream" >> "$shots/carla_frame_grid.log"
  "$FFMPEG" -loglevel error -y -i "assets/videos/$clip.mp4" -vf \
    "select='eq(n\,0)+eq(n\,100)+eq(n\,200)+eq(n\,299)',scale=320:192:flags=area,tile=4x1:padding=4:color=white" \
    -fps_mode passthrough -frames:v 1 "$OUT/row_$clip.png" || continue
  args=(-annotate +5+22 "${clip#carla_}")
  i=0
  for n in 0 100 200 299; do
    args+=(-annotate "+$((i * 324 + 5))+5" "frame $n, t = $(python -c "print(f'{$n / 20:.2f}')") s")
    i=$((i + 1))
  done
  convert "$OUT/row_$clip.png" -font DejaVu-Sans -pointsize 11 -fill white -undercolor '#000000a0' -gravity NorthWest \
    "${args[@]}" "$OUT/row_$clip.png"
  rows+=("$OUT/row_$clip.png")
done
convert "${rows[@]}" -background white -splice 0x4 -append -bordercolor white -border 4 "$shots/carla_frame_grid.png"

# 3. Terminals on a virtual display.
export DISPLAY=:99
Xvfb :99 -screen 0 1700x1400x24 -nolisten tcp & sleep 2
XT=(-fa 'DejaVu Sans Mono' -fs 10 -bg '#1e1e1e' -fg '#d4d4d4')
term() { # name columns rows
  local name=$1 cols=$2 rows=$3 pid wid
  rm -f "$OUT/$name.log"
  xterm -geometry "${cols}x${rows}+0+0" "${XT[@]}" -T "$name" \
    -e bash -c "docs/capture/capture.sh step $name 2>&1 | tee build/capture/$name.log; echo '=== done' >> build/capture/$name.log; exec sleep 900" &
  pid=$!
  for _ in $(seq 1 300); do grep -q "^=== done" "$OUT/$name.log" 2>/dev/null && break; sleep 2; done; sleep 2
  wid=$(xdotool search --sync --name "^$name\$" | head -1)
  import -window "$wid" "$shots/terminal_$name.png"
  grep -v "^=== done" "$OUT/$name.log" > "$shots/terminal_$name.log"
  kill $pid 2>/dev/null || true; sleep 1
}
term pytest     150 58
term fw_host    150 14
term fw_m4_size 120 72
term fw_m4_qemu 150 40
term mac_table  150 40
term nvcc       150 36

ls -la "$shots"
