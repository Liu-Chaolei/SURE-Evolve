# Copyright (c) Opendatalab. All rights reserved.
import copy
import json
import os
from pathlib import Path

try:
    from loguru import logger
except ImportError:  # optional; only needed for richer MinerU diagnostics.
    import logging

    logger = logging.getLogger("mineru_utils")
from .gpu_utils import (
    get_preferred_device,
    get_cuda_visible_device_value,
    get_ranked_gpu_candidates,
)


_MINERU_BACKEND = None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _configure_mineru_runtime():
    ranked = get_ranked_gpu_candidates()
    preferred_device = get_preferred_device()
    min_free_gb = _env_int("SURVEY_AGENT_MINERU_MIN_FREE_GB", 12)
    min_free_bytes = min_free_gb * 1024 * 1024 * 1024

    selected_free_bytes = None
    for device, free_bytes in ranked:
        if device == preferred_device:
            selected_free_bytes = free_bytes
            break

    if preferred_device.startswith("cuda") and (
        selected_free_bytes is None or selected_free_bytes < min_free_bytes
    ):
        free_gb = 0.0 if selected_free_bytes is None else selected_free_bytes / (1024**3)
        logger.warning(
            "MinerU GPU parsing disabled because the best visible GPU does not have enough free memory: "
            f"{preferred_device} has {free_gb:.2f} GiB free, threshold is {min_free_gb} GiB. Falling back to CPU."
        )
        preferred_device = "cpu"

    # MinerU pipeline defaults are aggressive for shared GPUs; keep them conservative.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("MINERU_PDF_RENDER_THREADS", os.environ.get("SURVEY_AGENT_MINERU_PDF_RENDER_THREADS", "1"))
    os.environ.setdefault(
        "MINERU_MIN_BATCH_INFERENCE_SIZE",
        os.environ.get("SURVEY_AGENT_MINERU_MIN_BATCH_INFERENCE_SIZE", "64"),
    )

    visible_value = get_cuda_visible_device_value(preferred_device)
    if visible_value is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_value
        os.environ["MINERU_DEVICE_MODE"] = "cuda"
        logger.info(
            "Configured MinerU for GPU parsing: "
            f"{preferred_device} via CUDA_VISIBLE_DEVICES={visible_value}, "
            f"MINERU_PDF_RENDER_THREADS={os.environ['MINERU_PDF_RENDER_THREADS']}, "
            f"MINERU_MIN_BATCH_INFERENCE_SIZE={os.environ['MINERU_MIN_BATCH_INFERENCE_SIZE']}"
        )
    else:
        os.environ["MINERU_DEVICE_MODE"] = "cpu"
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        logger.info(
            "Configured MinerU for CPU parsing with "
            f"MINERU_PDF_RENDER_THREADS={os.environ['MINERU_PDF_RENDER_THREADS']} and "
            f"MINERU_MIN_BATCH_INFERENCE_SIZE={os.environ['MINERU_MIN_BATCH_INFERENCE_SIZE']}"
        )
    return preferred_device


def _patch_mineru_pipeline_runtime(pipeline_analyze_module, batch_analyze_module):
    original_get_vram = getattr(pipeline_analyze_module, "get_vram", None)

    def _free_vram_gb(device):
        if original_get_vram is None:
            return 0
        try:
            import torch

            if str(device).startswith("cuda") and torch.cuda.is_available():
                if ":" in str(device):
                    device_index = int(str(device).split(":", 1)[1])
                else:
                    device_index = torch.cuda.current_device()
                free_bytes, _total_bytes = torch.cuda.mem_get_info(device_index)
                return max(1, round(free_bytes / (1024**3)))
        except Exception:
            pass
        return original_get_vram(device)

    pipeline_analyze_module.get_vram = _free_vram_gb

    mfr_base_batch_size = _env_int("SURVEY_AGENT_MINERU_MFR_BASE_BATCH_SIZE", 2)
    if hasattr(batch_analyze_module, "MFR_BASE_BATCH_SIZE"):
        batch_analyze_module.MFR_BASE_BATCH_SIZE = max(1, mfr_base_batch_size)

    logger.info(
        "Patched MinerU pipeline runtime to use free VRAM for batch sizing and "
        f"MFR_BASE_BATCH_SIZE={getattr(batch_analyze_module, 'MFR_BASE_BATCH_SIZE', 'unknown')}"
    )


def _load_mineru_backend():
    global _MINERU_BACKEND
    if _MINERU_BACKEND is not None:
        return _MINERU_BACKEND

    preferred_device = _configure_mineru_runtime()

    import mineru.backend.pipeline.batch_analyze as pipeline_batch_analyze
    from mineru.backend.pipeline.model_json_to_middle_json import (
        result_to_middle_json as pipeline_result_to_middle_json,
    )
    from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
    import mineru.backend.pipeline.pipeline_analyze as pipeline_analyze_module
    from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
        union_make as pipeline_union_make,
    )
    from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
    from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
    from mineru.cli.common import (
        convert_pdf_bytes_to_bytes_by_pypdfium2,
        prepare_env,
        read_fn,
    )
    from mineru.data.data_reader_writer import FileBasedDataWriter
    from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
    from mineru.utils.enum_class import MakeMode
    from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path

    _patch_mineru_pipeline_runtime(pipeline_analyze_module, pipeline_batch_analyze)

    _MINERU_BACKEND = {
        "FileBasedDataWriter": FileBasedDataWriter,
        "MakeMode": MakeMode,
        "convert_pdf_bytes_to_bytes_by_pypdfium2": convert_pdf_bytes_to_bytes_by_pypdfium2,
        "draw_layout_bbox": draw_layout_bbox,
        "draw_span_bbox": draw_span_bbox,
        "guess_suffix_by_path": guess_suffix_by_path,
        "pipeline_doc_analyze": pipeline_doc_analyze,
        "pipeline_result_to_middle_json": pipeline_result_to_middle_json,
        "pipeline_union_make": pipeline_union_make,
        "prepare_env": prepare_env,
        "read_fn": read_fn,
        "vlm_doc_analyze": vlm_doc_analyze,
        "vlm_union_make": vlm_union_make,
    }
    return _MINERU_BACKEND


def do_parse(
    output_dir,  # Output directory for storing parsing results
    pdf_file_names: list[str],  # List of PDF file names to be parsed
    pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
    p_lang_list: list[str],  # List of languages for each PDF, default is 'ch' (Chinese)
    backend="pipeline",  # The backend for parsing PDF, default is 'pipeline'
    parse_method="auto",  # The method for parsing PDF, default is 'auto'
    formula_enable=True,  # Enable formula parsing
    table_enable=True,  # Enable table parsing
    server_url=None,  # Server URL for vlm-http-client backend
    f_draw_layout_bbox=True,  # Whether to draw layout bounding boxes
    f_draw_span_bbox=True,  # Whether to draw span bounding boxes
    f_dump_md=True,  # Whether to dump markdown files
    f_dump_middle_json=True,  # Whether to dump middle JSON files
    f_dump_model_output=True,  # Whether to dump model output files
    f_dump_orig_pdf=True,  # Whether to dump original PDF files
    f_dump_content_list=True,  # Whether to dump content list files
    f_make_md_mode=None,  # The mode for making markdown content, default is MM_MD
    start_page_id=0,  # Start page ID for parsing, default is 0
    end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
):
    mineru = _load_mineru_backend()
    convert_pdf_bytes_to_bytes_by_pypdfium2 = mineru["convert_pdf_bytes_to_bytes_by_pypdfium2"]
    FileBasedDataWriter = mineru["FileBasedDataWriter"]
    MakeMode = mineru["MakeMode"]
    pipeline_doc_analyze = mineru["pipeline_doc_analyze"]
    pipeline_result_to_middle_json = mineru["pipeline_result_to_middle_json"]
    prepare_env = mineru["prepare_env"]
    vlm_doc_analyze = mineru["vlm_doc_analyze"]
    if f_make_md_mode is None:
        f_make_md_mode = MakeMode.MM_MD

    if backend == "pipeline":
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
                pdf_bytes, start_page_id, end_page_id
            )
            pdf_bytes_list[idx] = new_pdf_bytes

        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
            pipeline_doc_analyze(
                pdf_bytes_list,
                p_lang_list,
                parse_method=parse_method,
                formula_enable=formula_enable,
                table_enable=table_enable,
            )
        )

        for idx, model_list in enumerate(infer_results):
            model_json = copy.deepcopy(model_list)
            pdf_file_name = pdf_file_names[idx]
            local_image_dir, local_md_dir = prepare_env(
                output_dir, pdf_file_name, parse_method
            )
            image_writer, md_writer = FileBasedDataWriter(
                local_image_dir
            ), FileBasedDataWriter(local_md_dir)

            images_list = all_image_lists[idx]
            pdf_doc = all_pdf_docs[idx]
            _lang = lang_list[idx]
            _ocr_enable = ocr_enabled_list[idx]
            middle_json = pipeline_result_to_middle_json(
                model_list,
                images_list,
                pdf_doc,
                image_writer,
                _lang,
                _ocr_enable,
                formula_enable,
            )

            pdf_info = middle_json["pdf_info"]

            pdf_bytes = pdf_bytes_list[idx]
            _process_output(
                pdf_info,
                pdf_bytes,
                pdf_file_name,
                local_md_dir,
                local_image_dir,
                md_writer,
                f_draw_layout_bbox,
                f_draw_span_bbox,
                f_dump_orig_pdf,
                f_dump_md,
                f_dump_content_list,
                f_dump_middle_json,
                f_dump_model_output,
                f_make_md_mode,
                middle_json,
                model_json,
                is_pipeline=True,
            )
    else:
        if backend.startswith("vlm-"):
            backend = backend[4:]

        f_draw_span_bbox = False
        parse_method = "vlm"
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            pdf_file_name = pdf_file_names[idx]
            pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
                pdf_bytes, start_page_id, end_page_id
            )
            local_image_dir, local_md_dir = prepare_env(
                output_dir, pdf_file_name, parse_method
            )
            image_writer, md_writer = FileBasedDataWriter(
                local_image_dir
            ), FileBasedDataWriter(local_md_dir)
            middle_json, infer_result = vlm_doc_analyze(
                pdf_bytes,
                image_writer=image_writer,
                backend=backend,
                server_url=server_url,
            )

            pdf_info = middle_json["pdf_info"]

            _process_output(
                pdf_info,
                pdf_bytes,
                pdf_file_name,
                local_md_dir,
                local_image_dir,
                md_writer,
                f_draw_layout_bbox,
                f_draw_span_bbox,
                f_dump_orig_pdf,
                f_dump_md,
                f_dump_content_list,
                f_dump_middle_json,
                f_dump_model_output,
                f_make_md_mode,
                middle_json,
                infer_result,
                is_pipeline=False,
            )


def _process_output(
    pdf_info,
    pdf_bytes,
    pdf_file_name,
    local_md_dir,
    local_image_dir,
    md_writer,
    f_draw_layout_bbox,
    f_draw_span_bbox,
    f_dump_orig_pdf,
    f_dump_md,
    f_dump_content_list,
    f_dump_middle_json,
    f_dump_model_output,
    f_make_md_mode,
    middle_json,
    model_output=None,
    is_pipeline=True,
):
    """处理输出文件"""
    mineru = _load_mineru_backend()
    draw_layout_bbox = mineru["draw_layout_bbox"]
    draw_span_bbox = mineru["draw_span_bbox"]
    MakeMode = mineru["MakeMode"]
    pipeline_union_make = mineru["pipeline_union_make"]
    vlm_union_make = mineru["vlm_union_make"]

    if f_draw_layout_bbox:
        draw_layout_bbox(
            pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf"
        )

    if f_draw_span_bbox:
        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

    if f_dump_orig_pdf:
        md_writer.write(
            f"{pdf_file_name}_origin.pdf",
            pdf_bytes,
        )

    image_dir = str(os.path.basename(local_image_dir))

    if f_dump_md:
        make_func = pipeline_union_make if is_pipeline else vlm_union_make
        md_content_str = make_func(pdf_info, f_make_md_mode, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}.md",
            md_content_str,
        )

    if f_dump_content_list:
        make_func = pipeline_union_make if is_pipeline else vlm_union_make
        content_list = make_func(pdf_info, MakeMode.CONTENT_LIST, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4),
        )

    if f_dump_middle_json:
        md_writer.write_string(
            f"{pdf_file_name}_middle.json",
            json.dumps(middle_json, ensure_ascii=False, indent=4),
        )

    if f_dump_model_output:
        md_writer.write_string(
            f"{pdf_file_name}_model.json",
            json.dumps(model_output, ensure_ascii=False, indent=4),
        )

    logger.info(f"local output dir is {local_md_dir}")


def parse_doc(
    path_list: list[Path],
    output_dir,
    lang="ch",
    backend="pipeline",
    method="auto",
    server_url=None,
    start_page_id=0,
    end_page_id=None,
):
    """
    Parameter description:
    path_list: List of document paths to be parsed, can be PDF or image files.
    output_dir: Output directory for storing parsing results.
    lang: Language option, default is 'ch', optional values include['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']。
        Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
        Adapted only for the case where the backend is set to "pipeline"
    backend: the backend for parsing pdf:
        pipeline: More general.
        vlm-transformers: More general.
        vlm-vllm-engine: Faster(engine).
        vlm-http-client: Faster(client).
        without method specified, pipeline will be used by default.
    method: the method for parsing pdf:
        auto: Automatically determine the method based on the file type.
        txt: Use text extraction method.
        ocr: Use OCR method for image-based PDFs.
        Without method specified, 'auto' will be used by default.
        Adapted only for the case where the backend is set to "pipeline".
    server_url: When the backend is `http-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
    start_page_id: Start page ID for parsing, default is 0
    end_page_id: End page ID for parsing, default is None (parse all pages until the end of the document)
    """
    try:
        mineru = _load_mineru_backend()
        read_fn = mineru["read_fn"]
        file_name_list = []
        pdf_bytes_list = []
        lang_list = []
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
            lang_list.append(lang)
        do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=lang_list,
            backend=backend,
            parse_method=method,
            server_url=server_url,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
        )
    except Exception as e:
        logger.exception(f"Error in mineru parse_doc: {e}")


if __name__ == "__main__":
    # For test purpose
    from pathlib import Path

    base_dir = Path(os.environ.get("XLAB_SURVEY_AGENT_BASE_DIR", "."))
    test_pdf_path = base_dir / "legacy_workspace" / "data" / "sample_papers" / "paper_pdf" / "test.pdf"
    parse_doc(
        path_list=[test_pdf_path],
        output_dir=str(base_dir / "output" / "test_parse_doc"),
        lang="ch",
        backend="pipeline",
        method="auto",
    )
