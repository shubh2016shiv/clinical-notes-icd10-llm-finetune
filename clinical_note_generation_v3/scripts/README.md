# Clinical Note Generation v3 Scripts

Run these commands from the project root:

```powershell
cd "D:\Generative AI Portfolio Projects\clinical_notes_ICD_10_finetuning"
```

## 1. Build A Fresh FAISS ICD-10 Index

The generation pipeline requires mandatory hybrid retrieval. Build the FAISS
index before generating notes:

```powershell
python clinical_note_generation_v3\scripts\generate_FAISS_ICD10_index.py --reset-existing
```

Smoke test with fewer records:

```powershell
python clinical_note_generation_v3\scripts\generate_FAISS_ICD10_index.py --reset-existing --limit-records 100
```

Useful indexing options:

```powershell
python clinical_note_generation_v3\scripts\generate_FAISS_ICD10_index.py --reset-existing --batch-size 16
python clinical_note_generation_v3\scripts\generate_FAISS_ICD10_index.py --reset-existing --persist-dir C:\tmp\clinical_v3_faiss
python clinical_note_generation_v3\scripts\generate_FAISS_ICD10_index.py --reset-existing --no-prefer-gpu
```

Indexing uses OpenAI embeddings only. The provider/model is written to the
FAISS manifest, and clinical note generation uses that exact OpenAI model for
retrieval.

## 2. Generate Clinical Notes

Generate five notes:

```powershell
python clinical_note_generation_v3\scripts\generate_clinical_notes.py --count 5
```

Generate one note for an end-to-end smoke test:

```powershell
python clinical_note_generation_v3\scripts\generate_clinical_notes.py --count 1
```

Generate notes into a custom output directory:

```powershell
python clinical_note_generation_v3\scripts\generate_clinical_notes.py --count 5 --output-dir .\clinical_note_generation_v3\sample_data\manual_run
```

Skip artifact writes:

```powershell
python clinical_note_generation_v3\scripts\generate_clinical_notes.py --count 5 --no-artifacts
```

## Required Configuration

Put keys in the project-root `.env` or process environment:

```powershell
OPENAI_API_KEY=...
DEEPSEEK_API_KEY=...
```

Useful v3 knobs:

```powershell
CLINICAL_V3_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CLINICAL_V3_OPENAI_GENERATION_MODEL=gpt-4o-mini
CLINICAL_V3_DEEPSEEK_MODEL=deepseek-v4-flash
CLINICAL_V3_DEEPSEEK_BASE_URL=https://api.deepseek.com
CLINICAL_V3_FAISS_INDEX_BATCH_SIZE=16
CLINICAL_V3_FAISS_PERSIST_DIRECTORY=clinical_note_generation_v3\.faiss\icd10cm_2026_april_1
```

Gemini settings and clients remain in the codebase for future experiments, but
the default v3 runtime does not call Gemini.

## What To Expect

The FAISS build prints the OpenAI embedding model selected for indexing. It
writes vectors, ICD metadata, and a manifest into `.faiss`.

The note generation command prints per-note evaluation metrics:

- template id and archetype
- deterministic precheck result
- support verifier result
- general quality score
- ICD alignment score
- combined score
- final decision and revision count

If generation fails with a hybrid retrieval message, rebuild the index:

```powershell
python clinical_note_generation_v3\scripts\generate_FAISS_ICD10_index.py --reset-existing
```
