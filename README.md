# CNNs
Convolutional Neural Networks

As Convolutional Neural Networks (CNNs) são um tipo de rede neural especialmente projetada para lidar com dados que possuem uma estrutura de grade, como imagens e séries temporais. Elas são amplamente utilizadas em tarefas de visão computacional, como classificação de imagens, detecção de objetos e segmentação de imagens, devido à sua habilidade de extrair automaticamente características significativas diretamente dos dados brutos.

---

### Estrutura Básica de uma CNN

Uma CNN é composta por várias camadas organizadas para aprender representações hierárquicas dos dados de entrada:

1. **Camadas de Convolução (Convolutional Layers):**  
   - Executam operações de convolução sobre os dados de entrada usando filtros (ou kernels).  
   - Esses filtros são matrizes pequenas que percorrem a entrada para detectar padrões locais, como bordas, texturas ou formas específicas.  
   - Cada filtro aprende a identificar um padrão distinto nos dados.

2. **Camadas de Pooling (Pooling Layers):**  
   - Reduzem a dimensionalidade dos dados, simplificando a representação e tornando o modelo mais eficiente.  
   - O pooling ajuda a manter características importantes, como a posição relativa dos padrões, enquanto descarta detalhes irrelevantes.  
   - Os métodos mais comuns incluem o **max pooling** (seleciona o valor máximo em uma região) e o **average pooling** (calcula a média dos valores em uma região).

3. **Camadas de Ativação:**  
   - Aplicam funções não lineares, como a ReLU (Rectified Linear Unit), para introduzir não linearidades e permitir que a rede aprenda representações complexas.

4. **Camadas Fully Connected (Totalmente Conectadas):**  
   - São utilizadas nas etapas finais da CNN.  
   - Conectam todos os neurônios da camada anterior, produzindo as previsões finais para a tarefa, como a classificação em categorias.

5. **Camadas de Normalização e Dropout (Opcional):**  
   - **Normalização** ajuda a estabilizar e acelerar o treinamento.  
   - **Dropout** reduz o risco de overfitting, desativando aleatoriamente neurônios durante o treinamento.

---

### Principais Vantagens das CNNs
1. **Autonomia na Extração de Características:**  
   CNNs eliminam a necessidade de extrair manualmente características das imagens, já que aprendem essas características automaticamente durante o treinamento.

2. **Redução de Parâmetros:**  
   Comparadas a redes totalmente conectadas, as CNNs reduzem significativamente o número de parâmetros, aproveitando a estrutura local dos dados.

3. **Invariância Espacial:**  
   Graças às operações de convolução e pooling, as CNNs conseguem reconhecer padrões independentemente de sua posição na imagem.

---

### Aplicações das CNNs
1. **Classificação de Imagens:** Identificação de objetos em imagens, como "gato", "carro" ou "casa".  
2. **Detecção de Objetos:** Localização de múltiplos objetos em uma imagem com suas posições exatas.  
3. **Segmentação de Imagens:** Identificação de quais pixels pertencem a determinados objetos.  
4. **Reconhecimento Facial:** Identificação e verificação de rostos humanos.  
5. **Diagnóstico Médico:** Análise de imagens médicas, como radiografias e tomografias.

---

### Resumo
As CNNs revolucionaram a área de aprendizado de máquina, tornando possíveis avanços significativos em aplicações baseadas em imagens e vídeos. Elas aproveitam a ideia de convoluções para aprender padrões de maneira eficiente, resultando em modelos poderosos e escaláveis.
