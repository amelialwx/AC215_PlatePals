rm -f effnetv2b0-trainer.tar effnetv2b0-trainer.tar.gz
tar cvf effnetv2b0-trainer.tar package
gzip effnetv2b0-trainer.tar
gsutil cp effnetv2b0-trainer.tar.gz $GCS_BUCKET_URI/platepals-effnetv2b0-trainer.tar.gz