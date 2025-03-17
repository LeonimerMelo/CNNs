'''
Arquitetura CNN: Uma rede convolucional com 3 blocos convolucionais seguidos de camadas 
totalmente conectadas.

Função de treinamento: Implementada com monitoramento das métricas de perda e acurácia 
para treinamento e validação.

Gráficos de perdas e acurácia: Plotagem das curvas de treinamento e validação ao longo 
das épocas.

Função de avaliação: Avaliação do modelo no conjunto de teste com cálculo de métricas 
detalhadas.

Gráficos e tabelas de métricas: Matriz de confusão, relatório de classificação e 
gráficos de barras para métricas por classe.

Função para avaliação em imagens da internet: 
    Carrega e pré-processa imagens de URLs
    Classifica as imagens mostrando a classe prevista
    Mede o tempo de inferência
    Gera estatísticas e visualizações das métricas de tempo

Função principal: Que executa todo o pipeline de treinamento, avaliação e teste.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

path = 'C:\\Leo\\python scripts\\'

# Definição de device para usar GPU se disponível
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Utilizando device: {device}")

# Classes do CIFAR-10
classes = ('avião', 'automóvel', 'pássaro', 'gato', 'veado', 
           'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão')

# Transformações para normalização das imagens
# Converte para tensor e normaliza com média e desvio padrão calculados do CIFAR-10
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Aplicando data augmentation com crop aleatório
    transforms.RandomHorizontalFlip(),     # Aplicando data augmentation com flip horizontal
    transforms.ToTensor(),                 # Converte para tensor (valores entre 0-1)
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Normalização com média e desvio padrão
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# Carregamento dos dados de treinamento, validação e teste
def load_data(batch_size=128):
    """
    Carrega e prepara os datasets CIFAR-10 para treinamento, validação e teste
    
    Args:
        batch_size (int): Tamanho do batch para os dataloaders
        
    Returns:
        dataloaders: Dicionário com os dataloaders para train, val e test
    """
    # Carregando dataset de treinamento completo
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform_train)
    
    # Dividindo em treinamento (80%) e validação (20%)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])
    
    # Carregando dataset de teste
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform_test)
    
    # Criando os dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                           shuffle=False, num_workers=2)
    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    return dataloaders

# Definição da arquitetura da CNN
class CNN(nn.Module):
    def __init__(self):
        """
        Inicializa a arquitetura da rede neural convolucional
        """
        super(CNN, self).__init__()
        
        # Bloco 1: Convolução -> Batch Normalization -> ReLU -> MaxPooling
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Mantém a dimensão espacial
        self.bn1 = nn.BatchNorm2d(32)  # Normalização por batch para estabilizar o treinamento
        
        # Bloco 2: Convolução -> Batch Normalization -> ReLU -> MaxPooling
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Bloco 3: Convolução -> Batch Normalization -> ReLU -> MaxPooling
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling e funções de ativação (serão reusadas em blocos diferentes)
        self.pool = nn.MaxPool2d(2, 2)  # Reduz a dimensão espacial pela metade
        self.relu = nn.ReLU(inplace=True)  # Função de ativação ReLU
        
        # Dropout para regularização
        self.dropout = nn.Dropout(0.25)  # Desativa aleatoriamente 25% dos neurônios durante o treinamento
        
        # Camadas totalmente conectadas (fully connected)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Após 3 poolings: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)  # 10 classes para CIFAR-10
        
    def forward(self, x):
        """
        Método de forward propagation da rede
        
        Args:
            x: Entrada da rede (batch de imagens)
            
        Returns:
            Saída da rede (logits)
        """
        # Bloco 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        # Bloco 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Bloco 3
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Achatamento (flatten): de [batch, 128, 4, 4] para [batch, 128*4*4]
        x = x.view(-1, 128 * 4 * 4)
        
        # Camadas totalmente conectadas
        x = self.dropout(x)  # Primeiro dropout
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)  # Segundo dropout
        x = self.fc2(x)
        
        return x

# Função para treinar o modelo
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    """
    Treina o modelo e mostra a evolução de loss e acurácia
    
    Args:
        model (nn.Module): Modelo a ser treinado
        dataloaders (dict): Dicionário com dataloaders para 'train' e 'val'
        criterion: Função de perda a ser usada
        optimizer: Otimizador a ser usado
        num_epochs (int): Número de épocas de treinamento
        
    Returns:
        model: Modelo treinado
        history: Histórico de métricas durante o treinamento
    """
    since = time.time()
    
    # Inicializa listas para armazenar histórico de métricas
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Melhor acurácia de validação obtida
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Época {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Cada época tem uma fase de treinamento e validação
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Modo de treinamento
            else:
                model.eval()   # Modo de avaliação
                
            running_loss = 0.0
            running_corrects = 0
            
            # Itera sobre os dados
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zera os gradientes
                optimizer.zero_grad()
                
                # Forward
                # Rastreia histórico apenas no modo de treinamento
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + otimização apenas na fase de treinamento
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Estatísticas
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # Cálculo de métricas da época
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # Armazena métricas no histórico
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Salva o modelo se for o melhor na validação
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        
        print()
    
    # Tempo total
    time_elapsed = time.time() - since
    print(f'Treinamento completo em {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Melhor acurácia de validação: {best_acc:4f}')
    
    # Carrega os melhores pesos do modelo
    model.load_state_dict(best_model_wts)
    
    # Plota gráficos de perda e acurácia
    plot_training_history(history)
    
    return model, history

# Função para plotar o histórico de treinamento
def plot_training_history(history):
    """
    Plota gráficos de perda e acurácia durante o treinamento
    
    Args:
        history (dict): Dicionário com histórico de métricas
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de perda
    ax1.plot(history['train_loss'], label='Treino')
    ax1.plot(history['val_loss'], label='Validação')
    ax1.set_title('Perda')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Perda')
    ax1.legend()
    
    # Gráfico de acurácia
    ax2.plot(history['train_acc'], label='Treino')
    ax2.plot(history['val_acc'], label='Validação')
    ax2.set_title('Acurácia')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Acurácia')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Função para avaliar o modelo no conjunto de teste
def evaluate_model(model, dataloader):
    """
    Avalia o modelo no conjunto de teste
    
    Args:
        model (nn.Module): Modelo a ser avaliado
        dataloader: Dataloader do conjunto de teste
        
    Returns:
        float: Acurácia no conjunto de teste
        list: Lista de previsões
        list: Lista de labels verdadeiros
    """
    model.eval()
    
    # Listas para armazenar previsões e labels verdadeiros
    all_preds = []
    all_labels = []
    
    # Conta acertos
    correct = 0
    total = 0
    
    # Desativa cálculo de gradientes
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            # Armazena previsões e labels para métricas
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calcula acurácia final
    accuracy = 100 * correct / total
    print(f'Acurácia no conjunto de teste: {accuracy:.2f}%')
    
    return accuracy, all_preds, all_labels  
    
# Função para mostrar métricas detalhadas
def show_detailed_metrics(all_preds, all_labels):
    """
    Mostra métricas detalhadas e gráficos de avaliação
    
    Args:
        all_preds (list): Lista de previsões
        all_labels (list): Lista de labels verdadeiros
    """
    # Calcula a matriz de confusão
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plota a matriz de confusão
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.show()
    
    # Relatório de classificação
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    
    # Converte para DataFrame para melhor visualização
    df_report = pd.DataFrame(report).transpose()
    print("Relatório de Classificação:")
    print(df_report)
    
    # Plota gráfico de barras para precisão, recall e f1-score por classe
    plt.figure(figsize=(12, 6))
    df_report.iloc[:-3][['precision', 'recall', 'f1-score']].plot(kind='bar')
    plt.title('Métricas por Classe')
    plt.ylabel('Pontuação')
    plt.xlabel('Classe')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Calcula as métricas por classe
    class_metrics = pd.DataFrame({
        'Classe': classes,
        'Precisão': df_report.iloc[:-3]['precision'].values,
        'Recall': df_report.iloc[:-3]['recall'].values,
        'F1-Score': df_report.iloc[:-3]['f1-score'].values,
        'Suporte': df_report.iloc[:-3]['support'].values
    })
    
    print("\nMétricas por Classe:")
    print(class_metrics.to_string(index=False))

# Função para carregar e pré-processar imagens da internet
def load_and_preprocess_image(url):
    """
    Carrega uma imagem da internet e a pré-processa para entrada na rede
    
    Args:
        url (str): URL da imagem
        
    Returns:
        torch.Tensor: Tensor da imagem pré-processada
    """
    # Carrega a imagem da URL
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    
    # Redimensiona para 32x32 (tamanho do CIFAR-10)
    img = img.resize((32, 32), Image.LANCZOS)
    
    # Aplica as mesmas transformações usadas no conjunto de teste
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    # Converte para tensor e adiciona dimensão de batch
    img_tensor = transform(img).unsqueeze(0)
    
    return img_tensor, img

# Função para classificar uma imagem da internet
def classify_image(model, url):
    """
    Classifica uma imagem da internet e mostra a classificação e tempo de inferência
    
    Args:
        model (nn.Module): Modelo treinado
        url (str): URL da imagem
        
    Returns:
        int: Classe prevista
        float: Probabilidade da classe prevista
        float: Tempo de inferência (ms)
    """
    # Carrega e pré-processa a imagem
    img_tensor, original_img = load_and_preprocess_image(url)
    img_tensor = img_tensor.to(device)
    
    # Coloca o modelo em modo de avaliação
    model.eval()
    
    # Mede o tempo de inferência
    start_time = time.time()
    
    # Faz a inferência
    with torch.no_grad():
        # Executa a inferência 10 vezes para obter média mais precisa
        for _ in range(10):
            _ = model(img_tensor)
        
        # Agora faz a inferência real e mede o tempo
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        outputs = model(img_tensor)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
    
    # Calcula o tempo de inferência em milissegundos
    inference_time = (end_time - start_time) * 1000
    
    # Obtém a classe com maior probabilidade
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    pred_class = torch.argmax(probabilities, dim=1).item()
    pred_probability = probabilities[0][pred_class].item() * 100
    
    # Mostra a imagem original e a classificação
    plt.figure(figsize=(6, 6))
    plt.imshow(original_img)
    plt.axis('off')
    plt.title(f'Classificação: {classes[pred_class]} ({pred_probability:.2f}%)\nTempo: {inference_time:.2f} ms')
    plt.show()
    
    return pred_class, pred_probability, inference_time

# Função para avaliar múltiplas imagens da internet
def evaluate_internet_images(model, urls):
    """
    Avalia o modelo em múltiplas imagens da internet
    
    Args:
        model (nn.Module): Modelo treinado
        urls (list): Lista de URLs das imagens
    """
    # Resultados para cada imagem
    results = []
    
    for i, url in enumerate(urls):
        print(f"Classificando imagem {i+1}/{len(urls)}")
        try:
            pred_class, pred_probability, inference_time = classify_image(model, url)
            results.append({
                'URL': url,
                'Classe Prevista': classes[pred_class],
                'Probabilidade (%)': pred_probability,
                'Tempo de Inferência (ms)': inference_time
            })
        except Exception as e:
            print(f"Erro ao processar a imagem {i+1}: {str(e)}")
    
    # Cria um DataFrame com os resultados
    df_results = pd.DataFrame(results)
    
    # Mostra os resultados em uma tabela
    print("\nResultados da Classificação:")
    print(df_results.to_string(index=False))
    
    # Estatísticas de tempo de inferência
    if len(df_results) > 0:
        avg_time = df_results['Tempo de Inferência (ms)'].mean()
        min_time = df_results['Tempo de Inferência (ms)'].min()
        max_time = df_results['Tempo de Inferência (ms)'].max()
        
        print(f"\nEstatísticas de Tempo de Inferência:")
        print(f"Média: {avg_time:.2f} ms")
        print(f"Mínimo: {min_time:.2f} ms")
        print(f"Máximo: {max_time:.2f} ms")
        
        # Plota histograma de tempos de inferência
        plt.figure(figsize=(10, 6))
        plt.hist(df_results['Tempo de Inferência (ms)'], bins=10, alpha=0.7, color='blue')
        plt.axvline(avg_time, color='red', linestyle='dashed', linewidth=2)
        plt.title('Distribuição do Tempo de Inferência')
        plt.xlabel('Tempo (ms)')
        plt.ylabel('Frequência')
        plt.grid(True, alpha=0.3)
        plt.show()

# Função principal para executar o experimento completo
def run_experiment(num_epochs=10, batch_size=128, learning_rate=0.001):
    """
    Executa o experimento completo de treinamento e avaliação
    
    Args:
        num_epochs (int): Número de épocas para treinar
        batch_size (int): Tamanho do batch
        learning_rate (float): Taxa de aprendizado
    """
    print("Iniciando experimento de classificação de imagens CIFAR-10 com CNN")
    print(f"Device: {device}, Épocas: {num_epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
    
    # Carrega os dados
    print("\nCarregando dados...")
    dataloaders = load_data(batch_size)
    
    # Inicializa o modelo
    print("\nInicializando modelo...")
    model = CNN().to(device)
    
    # Sumário do modelo
    print("\nArquitetura do modelo:")
    print(model)
    
    # Mostra total de parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total de parâmetros: {total_params:,}")
    
    # Definindo função de perda e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Treina o modelo
    print("\nIniciando treinamento...")
    model, history = train_model(model, dataloaders, criterion, optimizer, num_epochs)
    
    # Avalia o modelo no conjunto de teste
    print("\nAvaliando modelo no conjunto de teste...")
    accuracy, all_preds, all_labels = evaluate_model(model, dataloaders['test'])
    
    # Mostra métricas detalhadas
    print("\nMostrando métricas detalhadas...")
    show_detailed_metrics(all_preds, all_labels)
    
    # Salva o modelo treinado
    print("\nSalvando modelo...")
    torch.save(model.state_dict(), path+'cifar10_cnn_model.pth')
    
    # Avalia em imagens da internet
    print("\nDeseja avaliar o modelo em imagens da internet? (S/N)")
    evaluate_internet = input().strip().lower() == 's'
    
    if evaluate_internet:
        print("Insira as URLs das imagens (uma por linha, digite 'fim' para terminar):")
        urls = []
        while True:
            url = input()
            if url.lower() == 'fim':
                break
            urls.append(url)
        
        if urls:
            print("\nAvaliando imagens da internet...")
            evaluate_internet_images(model, urls)
    
    print("\nExperimento concluído!")




# Executar o experimento com os parâmetros padrão (10 épocas, batch_size=128, learning_rate=0.001)
run_experiment()

# Ou você pode personalizar os parâmetros
# run_experiment(num_epochs=15, batch_size=64, learning_rate=0.0005)

# Exemplo de URLs para testar o modelo em imagens da internet
# URLs de exemplo para cada classe CIFAR-10:
urls = [
    "https://t4.ftcdn.net/jpg/09/54/87/95/360_F_954879570_fkOVusxxpQ4Qk0WFzdtVPGkbVlEmHDYo.jpg",
    #"https://example.com/airplane.jpg",      # Avião
    "https://img.freepik.com/vetores-gratis/carro-esportivo-azul-isolado-no-branco-vector_53876-67354.jpg",
    #"https://example.com/automobile.jpg",    # Automóvel
    # "https://example.com/bird.jpg",          # Pássaro
    "https://png.pngtree.com/png-clipart/20201117/ourmid/pngtree-bird-watercolor-hand-paint-png-image_2456682.jpg",
    # "https://example.com/cat.jpg",           # Gato
    "https://img.cdndsgni.com/preview/10094828.jpg",
    # "https://example.com/deer.jpg",          # 
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Veado-campeiro_macho_no_Parque_Nacional_da_Serra_da_Canastra_alt.jpg/2560px-Veado-campeiro_macho_no_Parque_Nacional_da_Serra_da_Canastra_alt.jpg",
    # "https://example.com/dog.jpg",           # 
    "https://www.pedigree.com.br/cdn-cgi/image/format=auto,q=90/sites/g/files/fnmzdf2401/files/2022-04/hero-icon-Savannah_0.png",
    # "https://example.com/frog.jpg",          # 
    "https://t3.ftcdn.net/jpg/00/52/79/80/360_F_52798076_6wr88EhTRTpZk8Mf69uXQnQDyotHGmrD.jpg",
    # "https://example.com/horse.jpg",         # 
    "https://dbd5813aed.cbaul-cdnwnd.com/043adc93ce56145d83140b16441e737c/system_preview_detail_200000012-bed28bf4e5-public/cavalo.jpg",
    # "https://example.com/ship.jpg",          # 
    "https://www.gov.br/transportes/pt-br/centrais-de-conteudo/navio-jpeg/@@images/image.jpeg",
    # "https://example.com/truck.jpg"          # 
    "https://img.freepik.com/psd-gratuitas/modelo-de-caminhao-caixa-psd_1409-3681.jpg"
]
# Nota: Substitua as URLs acima por URLs reais de imagens na internet
# que correspondam às classes do CIFAR-10

model = CNN().to(device)
model.load_state_dict(torch.load(path+'cifar10_cnn_model.pth', weights_only=True))
print("Loaded PyTorch Model State from model.pth")

evaluate_internet_images(model, urls)
