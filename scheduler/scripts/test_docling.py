#!/usr/bin/env python3
"""
Script: convert_pdf_to_json.py
Deskripsi: Convert PDF ke JSON menggunakan Docling di macOS (tanpa GPU)
Requirement: pip install docling
"""

import json
import sys
import time
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend


def buat_pipeline_options_cpu() -> PdfPipelineOptions:
    """
    Konfigurasi pipeline Docling khusus CPU (tanpa GPU).
    Menonaktifkan model AI berat yang membutuhkan GPU.
    """
    pipeline_options = PdfPipelineOptions()

    # --- Nonaktifkan model yang butuh GPU ---
    pipeline_options.do_table_structure = False   # TableFormer (butuh GPU/berat)
    pipeline_options.do_ocr = False               # OCR (aktifkan jika PDF hasil scan)

    # Gunakan backend ringan berbasis PDF native
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False

    return pipeline_options


def convert_pdf_ke_json(
    path_pdf: str | Path,
    path_output: str | Path | None = None,
    aktifkan_ocr: bool = False,
    engine_ocr: str = "easyocr",
) -> dict:
    """
    Convert file PDF ke format JSON menggunakan Docling.

    Args:
        path_pdf      : Path ke file PDF input
        path_output   : Path output JSON (opsional). Jika None, disimpan
                        di direktori yang sama dengan PDF.
        aktifkan_ocr  : True jika PDF hasil scan / berisi gambar teks
        engine_ocr    : 'easyocr' | 'tesseract' | 'tesseract_cli'

    Returns:
        dict: Hasil konversi dalam format JSON
    """
    path_pdf = Path(path_pdf)
    if not path_pdf.exists():
        raise FileNotFoundError(f"File PDF tidak ditemukan: {path_pdf}")

    print(f"📄 Input  : {path_pdf}")

    # --- Konfigurasi pipeline ---
    pipeline_options = buat_pipeline_options_cpu()

    if aktifkan_ocr:
        pipeline_options.do_ocr = True
        if engine_ocr == "easyocr":
            # EasyOCR: otomatis pakai CPU jika GPU tidak tersedia
            pipeline_options.ocr_options = EasyOcrOptions(force_full_page_ocr=False)
        elif engine_ocr == "tesseract":
            pipeline_options.ocr_options = TesseractOcrOptions()
        elif engine_ocr == "tesseract_cli":
            pipeline_options.ocr_options = TesseractCliOcrOptions()
        else:
            raise ValueError(f"engine_ocr tidak dikenal: {engine_ocr}")
        print(f"🔍 OCR    : Aktif ({engine_ocr})")
    else:
        print("🔍 OCR    : Nonaktif (PDF teks native)")

    # --- Buat converter ---
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,  # Backend ringan, tidak butuh GPU
            )
        }
    )

    # --- Konversi ---
    print("⏳ Mengkonversi...")
    mulai = time.time()
    hasil = converter.convert(str(path_pdf))
    selesai = time.time()
    print(f"✅ Selesai dalam {selesai - mulai:.2f} detik")

    # --- Export ke dict JSON ---
    dokumen = hasil.document
    data_json = dokumen.export_to_dict()

    # --- Simpan ke file ---
    if path_output is None:
        path_output = path_pdf.with_suffix(".json")
    else:
        path_output = Path(path_output)
        path_output.parent.mkdir(parents=True, exist_ok=True)

    with open(path_output, "w", encoding="utf-8") as f:
        json.dump(data_json, f, ensure_ascii=False, indent=2)

    print(f"💾 Output : {path_output}")
    print(f"📦 Ukuran : {path_output.stat().st_size / 1024:.1f} KB")

    return data_json


def convert_banyak_pdf(
    direktori_input: str | Path,
    direktori_output: str | Path | None = None,
    aktifkan_ocr: bool = False,
) -> list[dict]:
    """
    Convert semua file PDF dalam satu direktori ke JSON.

    Args:
        direktori_input  : Direktori berisi file PDF
        direktori_output : Direktori output JSON (opsional)
        aktifkan_ocr     : True jika PDF hasil scan

    Returns:
        list[dict]: Daftar hasil konversi
    """
    direktori_input = Path(direktori_input)
    file_pdf = list(direktori_input.glob("*.pdf"))

    if not file_pdf:
        print(f"⚠️  Tidak ada file PDF di: {direktori_input}")
        return []

    print(f"📂 Ditemukan {len(file_pdf)} file PDF\n")
    semua_hasil = []

    for i, pdf in enumerate(file_pdf, 1):
        print(f"[{i}/{len(file_pdf)}] Memproses: {pdf.name}")

        if direktori_output:
            output = Path(direktori_output) / pdf.with_suffix(".json").name
        else:
            output = None

        try:
            hasil = convert_pdf_ke_json(
                path_pdf=str(pdf),
                path_output=str(output) if output else None,
                aktifkan_ocr=aktifkan_ocr,
            )
            semua_hasil.append({"file": pdf.name, "status": "berhasil", "data": hasil})
        except Exception as e:
            print(f"❌ Gagal: {e}")
            semua_hasil.append({"file": pdf.name, "status": "gagal", "error": str(e)})

        print()

    berhasil = sum(1 for r in semua_hasil if r["status"] == "berhasil")
    print(f"📊 Ringkasan: {berhasil}/{len(file_pdf)} berhasil dikonversi")
    return semua_hasil


# =============================================================================
# CONTOH PENGGUNAAN
# =============================================================================

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # MODE 1: Convert satu file PDF
    # -------------------------------------------------------------------------
    if len(sys.argv) >= 2:
        path_input = sys.argv[1]
        path_output = sys.argv[2] if len(sys.argv) >= 3 else None

        hasil = convert_pdf_ke_json(
            path_pdf=path_input,
            path_output=path_output,
            aktifkan_ocr=False,   # Ganti True jika PDF hasil scan
            engine_ocr="easyocr", # Pilihan: 'easyocr', 'tesseract', 'tesseract_cli'
        )

        # Tampilkan preview JSON
        print("\n📋 Preview struktur JSON:")
        kunci = list(hasil.keys())
        print(f"   Kunci utama: {kunci}")

    # -------------------------------------------------------------------------
    # MODE 2: Contoh hardcode (ubah path sesuai kebutuhan)
    # -------------------------------------------------------------------------
    else:
        print("Penggunaan:")
        print("  python convert_pdf_to_json.py <input.pdf> [output.json]")
        print()
        print("Contoh:")
        print("  python convert_pdf_to_json.py dokumen.pdf")
        print("  python convert_pdf_to_json.py dokumen.pdf hasil/output.json")
        print()
        print("Untuk convert semua PDF dalam folder, edit bagian ini di kode:")
        print("  convert_banyak_pdf('folder_pdf/', 'folder_output/')")