set -e  # Exit on error

TAR_GLOB="/teamspace/gcs_folders/share/fmri-fm/datasets/hcp-parc/hcp-parc_*.tar"
OUTPUT_BASE="/teamspace/gcs_folders/share/fmri-fm/brain-jepa"
OUTPUT_DIR="${OUTPUT_BASE}/hcp-parc"
PARAMS_FILE="${OUTPUT_DIR}/normalization_params_hcp_train.npz"

ROI_COUNT_TOTAL=450
ROI_COUNT_KEEP=400
DROP_FIRST_ROIS=50

echo "=============================================================================="
echo "Step 1: Computing HCP normalization parameters"
echo "=============================================================================="

if [ -f "${PARAMS_FILE}" ]; then
    echo "Normalization params already exist: ${PARAMS_FILE}"
    echo "Delete it if you want to recompute."
else
    python src/brain_jepa/preprocessing/compute_hcp_normalization.py \
        --tar-glob "${TAR_GLOB}" \
        --output-dir "${OUTPUT_DIR}" \
        --roi-count-total ${ROI_COUNT_TOTAL} \
        --roi-count-keep ${ROI_COUNT_KEEP} \
        --drop-first-rois ${DROP_FIRST_ROIS}
fi

echo ""
echo "=============================================================================="
echo "Step 2: Extracting HCP timeseries to .pt files"
echo "=============================================================================="

python src/brain_jepa/preprocessing/extract_hcp_to_pt.py \
    --tar-glob "${TAR_GLOB}" \
    --output-dir "${OUTPUT_DIR}" \
    --roi-count-total ${ROI_COUNT_TOTAL} \
    --roi-count-keep ${ROI_COUNT_KEEP} \
    --drop-first-rois ${DROP_FIRST_ROIS} \
    --skip-normalization


echo ""
echo "=============================================================================="
echo "Step 3: Testing dataset loader"
echo "=============================================================================="

python src/brain_jepa/datasets/hcp_parc_pt.py "${OUTPUT_DIR}"

echo ""
echo "=============================================================================="
echo "✓ Extraction pipeline complete!"
echo "=============================================================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Normalization params: ${PARAMS_FILE}"
echo ""
echo "To use in training, update your config:"
echo "  data.pt_dir: ${OUTPUT_DIR}"
echo "=============================================================================="

