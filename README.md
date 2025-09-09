# Proyecto completo WDN

## Estructura
- src/simulate.py → genera datasets sintéticos a partir de YAML
- training/train.py → entrena el modelo MLP
- training/infer.py → realiza predicciones en nuevos datos
- configs/ctown_example.yaml → configuración de ejemplo
- data/processed/training_set_mini_demo.npz → dataset miniatura de prueba

## Flujo de uso
1. Simulación:
   python -m src.simulate --config configs/ctown_example.yaml

2. Entrenamiento:
   python -m training.train --npz data/processed/training_set_ctown_demo.npz --outdir artifacts_p_demo --mode p --x_names "JUNCTION-101,JUNCTION-205"

3. Inferencia:
   python -m training.infer --model artifacts_p_demo/best_model.pt --scalers artifacts_p_demo/scalers.npz --feature_names artifacts_p_demo/x_feature_names.json --mode p --csv data/processed/timeseries_ctown_demo.csv --dout 2 --map --save results/p_pred.csv
