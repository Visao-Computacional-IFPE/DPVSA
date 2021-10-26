# COMPUTER VISION CHALLENGE: SOCCER MATCH MONITORING

Para que se possa executar o código, o usuário precisará instalar em sua máquina as seguintes bibliotecas:

### **1) REQUIREMENTS**:

* pip install python = 3.9.7

* pip install tensorflow

* pip install opencv-python

* pip install opencv-contrib-python

* pip install scipy

**OBS**: Com a instalação das bibliotecas, automaticamente irá instalar a sua versão mais recente, no qual foi utilizada para desenvolver a nossa solução.

**OBS**: Realize o donwload do arquivo contido no link: https://pjreddie.com/media/files/yolov3-spp.weights e coloque no caminho `Yolov3/YL/`

### **2) EXECUÇÃO DO ALGORITMO**:

Após a instalação de todas as bibliotecas contidas no item *requirements*, iremos instruir abaixo, as etapas que precisam ser confeccionadas antes de se rodar o algoritmo:

1. Criar uma pasta chamada `data`
2. Dentro desta pasta, crie um outra chamada `videos
3. Dentro da pasta `videos` insirá todos os vídeos de teste das partidas de futebol disponibilizados
4. Após isso, abra o arquivo `Reproducer.py` e passe o caminho de onde o vídeo que você que testar está localizado, dentro da  variável `VIDEO_PATH`. 

Exemplo: `VIDEO_PATH = "data/videos/video3.mp4"`

### **3) IDEIA DA SOLUÇÃO ENCONTRADA**:

Inicialmente a ideia era treinar uma rede neural para identificação /classificação das pessoas em campo e detecção da bola e em seguida inserir esses dados em um algoritmo de tracking, mas mudamos de ideia ao perceber que a classificação das pessoas pela rede neural treinada torna a mesma muito imprecisa.  

Após esse primeiro momento decidimos treinar a rede apenas para identificação de pessoas em campo e em seguida determinar "quem ela é" através de comparações, agrupando as 2 grupos com maiores semelhança (que seriam classificados como os 2 times) e para identificação do arbitro, seria apenas verificar qual pessoa está mais próxima da linha horizontal central da imagem, excluindo jogadores já agrupados. 

No inicio da montagem do código percebemos que não havia um bom método para comparar vestimentas das pessoas identificadas, então decidimos treinar uma rede "Siamese" como é chamada, para comparação de imagens, que seria a solução perfeita para nosso problema, então automatizamos para extrair os dados do dataset fornecido e treinamos uma rede neural para comparação de vestimentas dos jogadores.

Na criação do código partimos para uma estratégia que apelidamos de "analise do quadro inicial" basicamente consiste na obtenção do que seria um jogador do time 01, 02 e arbitro, para que esses dados fossem utilizados para caracterizar todas aas pessoas que fossem detectadas ao longo dos frames extraídos. Então utilizamos o Yolov3 para identificar pessoas no campo, depois pegamos a pessoa mais a esquerda do frame e chamamos de jogador do time 1, a mais a direita é chamada jogador do time 2, e o mais próximo da linha horizontal do frame que não parece com nenhum dos jogadores é caracterizada como arbitro, com isso, conseguimos obter as imagens básicas para caracterizar qualquer pessoa identificada. 

Após isso utilizamos um tracker de centroide no qual relacionamos o id do centroide com a comparação de sua vestimenta e assim o id do centroide carrega o grupo a qual ele pertence.

### **4) PROBLEMA**:

Além dos problemas de detecção, que pode gerar erros na determinação do arbitro e dos jogadores dos times, a imprecisão da comparação também gera erros bem significativos, mas o grande problema do/da solução foi o desempenho, a rede siamese consome muito desempenho pois precisa comparar pessoas demais e por conta disso o programa e por esse motivo é impossível rodar a detecção e tracking em tempo real.

### REFERÊNCIAS BIBLIOGRÁFICAS:
https://pjreddie.com/darknet/yolo/

https://www.pyimagesearch.com/2020/12/07/comparing-images-for-similarity-using-siamese-networks-keras-and-tensorflow/

https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/



