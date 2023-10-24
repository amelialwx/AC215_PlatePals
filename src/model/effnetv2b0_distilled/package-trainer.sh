rm -f effnetv2b0-distilled-trainer.tar effnetv2b0-distilled-trainer.tar.gz
tar cvf effnetv2b0-distilled-trainer.tar package
gzip effnetv2b0-distilled-trainer.tar
gsutil cp effnetv2b0-distilled-trainer.tar.gz $GCS_BUCKET_URI/platepals-effnetv2b0-distilled-trainer.tar.gz