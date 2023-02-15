# VisInspection
___

Este módulo realiza a contagem de objetos na tela em tempo real. Para melhores resultados, é importante utilizar um fundo neutro e sem texturas, iluminação adequada, e espaçar os objetos a uma distância mínima de 1cm.

<img src="/Images/sample_1.png">



## Como instalar
Na pasta raiz do projeto, execute:

```
pip install .
```

## Como usar

```
from VisInspec.utils import main

main(cameraIndex=0)

```

## Problema na instalação

Após a intalação deste projeto, pode ser que ao executá-lo, a seguinte mensagem de erro seja apresentada: **No module named 'cv2.aruco'**.

Para resolver esse erro, desinstale as bibliotecas **opencv-python** e **opencv-contrib-python**. Depois, instale a biblioteca **opencv-contrib-python**.

```
pip uninstall opencv-python
pip uninstall opencv-contrib-python
pip install opencv-contrib-python
```
