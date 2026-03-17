import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import DeviceProperties

triton_helpers.set_driver_to_gpu()


@triton_heuristics.pointwise(
    size_hints={"x": 4096},
    filename=__file__,
    triton_meta={
        "signature": {
            "in_ptr0": "*i64",
            "in_ptr1": "*i64",
            "in_ptr2": "*fp64",
            "in_ptr3": "*fp64",
            "out_ptr0": "*fp64",
            "xnumel": "i32",
            "XBLOCK": "constexpr",
        },
        "device": DeviceProperties(
            type="cuda",
            index=0,
            multi_processor_count=132,
            cc=90,
            major=9,
            regs_per_multiprocessor=65536,
            max_threads_per_multi_processor=2048,
            max_threads_per_block=1024,
            warp_size=32,
        ),
        "constants": {},
        "native_matmul": False,
        "enable_fp_fusion": True,
        "launch_pdl": False,
        "disable_ftz": False,
        "configs": [
            {
                (0,): [["tt.divisibility", 16]],
                (1,): [["tt.divisibility", 16]],
                (2,): [["tt.divisibility", 16]],
                (3,): [["tt.divisibility", 16]],
                (4,): [["tt.divisibility", 16]],
                (5,): [["tt.divisibility", 16]],
            }
        ],
    },
    inductor_meta={
        "grid_type": "Grid1D",
        "autotune_hints": set(),
        "kernel_name": "triton_poi_fused_add_clone_expand_index_scatter_add_unsqueeze_zeros_57",
        "mutated_arg_names": ["out_ptr0"],
        "optimize_mem": True,
        "no_x_dim": False,
        "atomic_add_found": True,
        "num_load": 3,
        "num_store": 1,
        "num_reduction": 0,
        "backend_hash": "58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7",
        "assert_indirect_indexing": True,
        "autotune_local_cache": True,
        "autotune_pointwise": True,
        "autotune_remote_cache": None,
        "force_disable_caches": False,
        "dynamic_scale_rblock": True,
        "max_autotune": False,
        "max_autotune_pointwise": False,
        "min_split_scan_rblock": 256,
        "spill_threshold": 16,
        "store_cubin": False,
        "deterministic": False,
        "force_filter_reduction_configs": False,
        "are_deterministic_algorithms_enabled": False,
        "tiling_scores": {"x": 10240},
    },
    min_elem_per_thread=0,
)
@triton.jit
def triton_poi_fused_add_clone_expand_index_scatter_add_unsqueeze_zeros_57(
    in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 10) % 4
    x0 = xindex % 10
    x2 = xindex // 40
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy="evict_last")
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy="evict_last")
    tmp9 = tl.load(in_ptr3 + (x3), xmask)
    tl.device_assert(((0 <= tmp0) & (tmp0 < 1)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 1")
    tmp3 = tl.full([XBLOCK], 14, tl.int32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp2 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp2)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 14)) | ~(xmask), "index out of bounds: 0 <= tmp6 < 14")
    tmp8 = tl.load(in_ptr2 + (x0 + 10 * tmp6 + 140 * x2), xmask)
    tmp10 = tmp8 + tmp9
    tl.atomic_add(out_ptr0 + (x0 + 10 * x2), tmp10, xmask, sem="relaxed")
