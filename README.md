# MOT

## Install

```bash
pip3 install -r requirements.txt
```



## Demo

```bash
python demo_track.py video --common.config-file ./config/detection/edgeformer/ssd_edgeformer_s.yaml --model.detection.pretrained ./pretrained_models/detection/checkpoint_ema_avg.pt --evaluation.detection.mode validation_set --evaluation.detection.resize-input-images --save_result
```

