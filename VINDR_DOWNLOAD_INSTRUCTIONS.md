# VinDr-Mammo Download Instructions

## Getting Access to VinDr-Mammo Dataset

You need to request access to the VinDr-Mammo dataset on PhysioNet before you can download it.

### Steps to Get Access:

1. **Visit the dataset page:**
   - Go to: https://physionet.org/content/vindr-mammo/1.0.0/

2. **Sign the Data Use Agreement (DUA):**
   - Click the "Request Access" button or "Files" tab
   - You'll be prompted to sign the DUA
   - Read and accept the terms

3. **Complete Credentialing (if needed):**
   - PhysioNet may require identity verification
   - Upload required documents (ID, proof of training completion, etc.)
   - Wait for approval (usually 1-3 business days)

4. **Verify Access:**
   - Once approved, you should see "You have been granted access" message
   - The "Files" tab should show downloadable content

### Current Status:

❌ **Access Denied (HTTP 403)**
- Your account `t-9dolab@uchicago.edu` does not have access yet
- Complete the steps above to get access

### Once You Have Access:

Run the test download script:
```bash
cd "C:\Users\HP\Downloads\project"
export PHYSIONET_USERNAME='t-9dolab@uchicago.edu'
export PHYSIONET_PASSWORD='1234&abcd@D'
python test_vindr_download.py
```

This will download 5 stratified files (2 malignant + 3 benign) for testing.

### Alternative: Manual Download Test

If you prefer to test manually:

1. Visit: https://physionet.org/files/vindr-mammo/1.0.0/breast-level_annotations.csv
2. If you can download it in your browser, you have access
3. Then run the Python script

### Files That Will Be Downloaded:

```
vindr-mammo-test/
├── metadata/
│   └── breast-level_annotations.csv  (~2 MB)
└── images/
    ├── {study_id_1}/
    │   └── {image_id_1}.dicom  (~10-20 MB each)
    ├── {study_id_2}/
    │   └── {image_id_2}.dicom
    ...
    (5 DICOM files total: 2 malignant + 3 benign)
```

### Dataset Information:

- **Total Size:** ~163 GB (5,000 exams, 20,000 images)
- **Test Download:** 5 files (~50-100 MB)
- **Full Stratified Download (1000 files):** ~5-10 GB
- **Format:** DICOM (Digital Imaging and Communications in Medicine)
- **Stratification:**
  - Malignant: BI-RADS 4, 5, 6
  - Benign: BI-RADS 1, 2
  - Excluded: BI-RADS 3 (probably benign)

---

## Troubleshooting

### Error: "403 Forbidden"
- **Cause:** No access to dataset
- **Solution:** Request access on PhysioNet (see steps above)

### Error: "401 Unauthorized"
- **Cause:** Invalid credentials
- **Solution:** Check username and password

### Error: "Connection timeout"
- **Cause:** Network issues or PhysioNet downtime
- **Solution:** Try again later, check internet connection

---

**Last tested:** 2026-01-11
**Status:** Waiting for PhysioNet access approval
