# Estimacija poze - SC2019, FTN-AI-LAB


## Članovi tima:
* SW16/2016, Milan Milovanović
* SW52/2016, Nemanja Janković
* SW69/2016, Mihailo Đokić


## Issue:
https://github.com/ftn-ai-lab/sc-2019-siit/issues/22

## Instalacija aplikacije:

* Klonirati repozitorijum, pocizionirati se u root folder
* Pokrenuti ```pip install -r requirements.txt ```
* Instalirati PyTorch prateći instrukcije sa https://pytorch.org/get-started/locally/
* Instalirati Nvidia CUDA biblioteku sa https://developer.nvidia.com/cuda-downloads
* Instalirati Nvidia CUDNN biblioteku sa https://developer.nvidia.com/cudnn

## Skidanje dataset-a:
* Sa http://human-pose.mpi-inf.mpg.de/#download skinuti slike i ekstraktovati ih u ```datasets/MPII/images```
* Za generisanje train, validacionog i test skupa pokrenuti ```datasets/MPII/mpii.py```

## Pokretanje aplikacije:
* Pokretanje treniranja:
  * Pokrenuti ```train.py``` uz unos sledećih argumenata:
    * ```-model```, mogući modeli su [ 'resnet', 'mobilenetv2', 'shufflenetv2' ]
    * ```-epochs```, broj epoha treniranja
    * ```-save_location```, lokacija snimanja modela tokom treninga
    * ```--input_size```, veličina inputa pri treniranju, default 224
    * ```--batch_size```, veličina minibatch-a pri treniranju, default 32
    * ```--lr```, learning rate, default 1e-3
    * ```--t7```, lokacija prethodno snimljenog modela za nastavak treniranja, default ""
* Pokretanje testiranja:
  * Pokrenuti ```test.py``` uz unos sledećih argumenata:
    * ```-model```, mogući modeli su [ 'resnet', 'mobilenetv2', 'shufflenetv2' ]
    * ```--input_size```, veličina inputa pri treniranju, default 224
    * ```--batch_size```, veličina minibatch-a pri treniranju, default 32
    * ```--t7```, lokacija prethodno snimljenog modela za nastavak treniranja, default ""
* Pokretanje evaluacije:
  * Pokrenuti ```test.py``` uz unos sledećih argumenata:
    * ```-model```, mogući modeli su [ 'resnet', 'mobilenetv2', 'shufflenetv2' ]
    * ```-t7```, lokacija prethodno snimljenog modela za nastavak treniranja 
    * ```--input_size```, veličina inputa pri treniranju, default 224
    * ```--batch_size```, veličina minibatch-a pri treniranju, default 32
* Pokretanje estimacije:
  * Pokrenuti ```predict_camera.py``` za predikciju slike sa web kamere uz unos sledećih argumenata:
    * ```-model```, mogući modeli su [ 'resnet', 'mobilenetv2', 'shufflenetv2' ]
    * ```-t7```, lokacija prethodno snimljenog modela za nastavak treniranja 
  * Pokrenuti ```predict_image.py``` za predikciju postojeće slike uz unos sledećih argumenata:
    * ```-model```, mogući modeli su [ 'resnet', 'mobilenetv2', 'shufflenetv2' ]
    * ```-t7```, lokacija prethodno snimljenog modela za nastavak treniranja 