# /bin/sh

curl http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz > nsynth-valid.jsonwav.tar.gz
curl http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz > nsynth-test.jsonwav.tar.gz
curl http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz > nsynth-train.jsonwav.tar.gz
tar -xzf nsynth-valid.jsonwav.tar.gz
tar -xzf nsynth-test.jsonwav.tar.gz
tar -xzf nsynth-train.jsonwav.tar.gz
rm nsynth-valid.jsonwav.tar.gz
rm nsynth-test.jsonwav.tar.gz
rm nsynth-train.jsonwav.tar.gz
