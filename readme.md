# Teoria

- [Chapter 1: Introduction](http://bit.ly/theMLbook-Chapter-1)
- [Chapter 2: Notation and Definitions](http://bit.ly/theMLbook-Chapter-2)
- [Chapter 3: Fundamental Algorithms](http://bit.ly/theMLbook-Chapter-3)
- [Chapter 4: Anatomy of a Learning Algorithm](http://bit.ly/theMLbook-Chapter-4)
- [Chapter 5: Basic Practice](http://bit.ly/theMLbook-Chapter-5)
- [Chapter 6: Neural Networks and Deep Learning](http://bit.ly/theMLbook-Chapter-6)
- [Chapter 7: Problems and Solutions](http://bit.ly/theMLbook-Chapter-7)
- [Chapter 8: Advanced Practice](http://bit.ly/theMLbook-Chapter-8)
- [Chapter 9: Unsupervised Learning](https://www.dropbox.com/s/y9a7b0hzmuksqar/Chapter9.pdf?dl=0)
- [Chapter 10: Other Forms of Learning](http://bit.ly/theMLbook-Chapter-10)
- [Chapter 11: Conclusion](http://bit.ly/theMLbook-Chapter-11)

# Cursos

- [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)

# Prática

- [Deep Learning with PyTorch for Medical Image Analysis](https://www.udemy.com/course/deep-learning-with-pytorch-for-medical-image-analysis/learn)

# Ferramentas

- [Tuning Playbook](https://github.com/google-research/tuning_playbook)

# Aprofundamento

- [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [Introduction to PyTorch - YouTube Series](https://pytorch.org/tutorials/beginner/introyt.html)
- Auto‐avalie o quanto você aprendeu dos assuntos tentando resolver um novo problema. A base de dados  [Chest  X‐Ray  Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) possui  um  conjunto  de  imagens  de  raio‐x  de  pessoas saudáveis e de pessoas com pneumonia. Realize fine‐tuning de uma CNN (sugestão: mobilenet‐v2) para classificar automaticamente as imagens. No processo, aprenda:

    - O que é o conjunto de validação, qual a sua diferença para os conjuntos de treino e teste e porque ele é necessário.

    - O que são hiperparâmetros, quais são os mais importantes no treinamento de redes neurais, como ajustá‐los usando o conjunto de validação e porque não podemos usar o conjunto de teste para avaliá‐lo.

    - O efeito de usar taxas de aprendizado muito grandes e muito pequenas. Compreender o porquê destes efeitos serem observados.

    - O que é uma matriz de confusão e as métricas de performance derivadas precision, recall e f1‐score. Em que situações elas são mais adequadas que a acurácia?

- [Médio] Selecione uma tarefa que você acha interessante no [papers with code](https://paperswithcode.com/sota). Ao clicar na tarefa, veja se ela possui um bechmark. Se sim, veja o modelo que aparece com mais frequência na lista. Entre no github oficial do projeto e tente baixar e usar o modelo pré‐treinado. Por exemplo, se você tiver interesse na tarefa de [análise de sentimentos em texto](https://paperswithcode.com/task/sentiment-analysis), tente baixar o modelo que alguém desenvolveu do github e usar o modelo para classificar uns textos que você digitar (note que se ele foi treinado com dados em inglês, você deve digitar textos em inglês). Como alguns modelos podem ser bem pesados, é recomendável usar o google colab. Também é uma boa ler o paper para ver se existem modelos de diferentes tamanhos e, se sim, dê preferência pelos menores porque os grandes podem não rodar nem no colab.

- [Fácil] Use a biblioteca gradio para criar uma interface web para demonstrar o seu modelo.

- [Fácil] Use as bibliotecas plotly e dash para criar dashboards web interativos para visualização de dados. Use bases de dados do kaggle para produzir novos dashboards.

- [Fácil] Use a biblioteca pytorch lightning para treinar redes neurais com pytorch usando menos linhas de código. Similar à biblioteca keras para tensorflow.

- [Médio] Compreenda como disponibilizar o modelo via um webservice usando flask, celery e redis. 

- [Fácil] Faça tutoriais de preparação de dados usando pandas. Faça a preparação usando uma nova base de dados do kaggle.

- [Fácil] Faça tutoriais de visualização de dados usando matplotlib e seaborn. Faça exemplos usando uma base de dados do kaggle.

- [Fácil e Muito Enriquecedor] Aprenda a usar a biblioteca scikit‐learn que tem implementações de vários  métodos  tradicionais  de  machine  learning.  Explorar  a  [página de exemplos](https://scikit-learn.org/stable/auto_examples/index.html) pode  ser  um caminho legal. Para cada método, estude‐o. O canal [StatQuest](https://www.youtube.com/@statquest) tem explicações MUITO didáticas dos métodos.

- [Médio, mais teórico e bom para entender em profunfidade o assunto] Existem diversos cursos no youtube que trazem conhecimentos profundos sobre machine learning e redes neurais. Por exemplo, os cursos do [prof. Andrew Ng](https://www.youtube.com/@Deeplearningai/playlists), do [prof. Yann LeCun](https://www.youtube.com/watch?v=0bMe_vCZo30&list=PL80I41oVxglKcAHllsU0txr3OuTTaWX2v) (vencedor do prêmio Turing por contribuições na área de [redes  neurais](https://www.youtube.com/watch?v=0bMe_vCZo30&list=PL80I41oVxglKcAHllsU0txr3OuTTaWX2v)),  do  [prof.  Geoffrey  Hinton](https://www.youtube.com/watch?v=OVwEeSsSCHE&list=PLLssT5z_DsK_gyrQ_biidwvPYCRNGI3iv)  (também  vencedor  do  prêmio  Turing  e, historicamente, [o pesquisador mais importante em redes neurais do mundo](https://www.youtube.com/watch?v=OVwEeSsSCHE&list=PLLssT5z_DsK_gyrQ_biidwvPYCRNGI3iv), na minha opinião) e do [prof. Andreas Geiger](https://www.youtube.com/watch?v=BHBAnUAdeyE&list=PL05umP7R6ij3NTWIdtMbfvX7Z-4WEXRqD) (um dos melhores pesquisadores dentre os mais jovens, na minha opinião, e uma [referência pessoal](https://www.youtube.com/watch?v=BHBAnUAdeyE&list=PL05umP7R6ij3NTWIdtMbfvX7Z-4WEXRqD)). É completamente inviável fazer todos os cursos, mas selecione um do professor que você for mais com a cara e siga nele. Sempre que possível, faça mini‐projetos que te permitam demonstrar o conhecimento adquirido nas aulas.

- [Difícil ‐ <span style="color:red">Quem fizer e de fato compreender tudo está contratado</span>] Implementar uma rede neural do tipo multilayer perceptron com número variável de camadas ocultas e neurônios por camada oculta e o algoritmo backpropagation sem usar o pytorch (mas podendo usar numpy). Para testar a rede, faça o exemplo de curve fitting abaixo usando ela ao invés do pytorch.

- [Médio] Aprenda a treinar um algoritmo de detecção de objetos usando o [tutorial do pytorch](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html). No processo, você deve compreender a diferença entre detecção de objetos e classificação de imagens, e as funções de perda e métricas de performance mais usadas na tarefa de detecção de objetos.

- [Médio] Use a biblioteca huggingface para fazer fine‐tuning de um vision transformer no problema de classificação de raios‐x entre saudáveis e com pneumonia. A menos que você tenha uma placa de vídeo com pelo menos 8GB de VRAM, será necessário usar o Google Colab.

- [Difícil]  Use  a  biblioteca  deeplab‐v2  para  resolver  um  tarefa  de  segmentação  de  imagens.  No processo, você deve compreender a diferença entre segmentação e classificação de imagens, e as funções de perda e métricas de performance mais usadas na tarefa de segmentação. Aproveite, e estude também o que são as tarefas de segmentação de instâncias e segmentação panóptica.

- [Médio para Difícil] Faça curve fitting usando uma rede neural do tipo multilayer perceptron usando pytorch. Para isto, gere pares (x, f(x)) para f(x) = sin(x) no intervalo entre [‐2PI, PI]. Separe parte dos pares para treinamento e parte para teste. Some valores amostrados de uma distribuição gaussiana (veja a função np.random.normal) aos valores de f(x) no conjunto de treinamento para simular ruído nas medições. Treine a rede neural para predizer o valor de f(x) a partir do valor de x. Use a função erro médio quadrático como função de perda e métrica de performance. Avalie a performance no conjunto de teste. Use a biblioteca matplotlib para gráficos (1) do MSE para os conjuntos de treino e teste durante o treinamento e (2) para visualizar os pontos reais e preditos pela rede neural em comparação com a função seno. Veja como a rede se comporta predizendo os valores de f(x) para x no intervalo de [2PI, 4PI]. Estude o que é interpolação e extrapolação, leia um pouco sobre a relação deles com redes neurais e tente explicar o efeito observado.