from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import numpy as np
import json
from .mlp import MLP

# Inicializa a MLP com 100 entradas, 50 neurônios ocultos e 5 saídas (apenas as letras que você enviar)
mlp = MLP(input_size=100, hidden_size=50, output_size=5)

class MLPTrainAPIView(APIView):
    def post(self, request, *args, **kwargs):
        try:
            # Receber os dados de entrada (5 matrizes 10x10 e labels)
            data = request.data.get('matrices')
            labels = request.data.get('labels')
            print(labels)

            # Validar que são 5 matrizes 10x10 com valores 0 ou 1
            if len(data) != 5 or any(len(matrix) != 10 or any(len(row) != 10 or any(cell not in [0, 1] for cell in row) for row in matrix) for matrix in data):
                return Response({"error": "Entrada deve ser 5 matrizes 10x10 com valores 0 ou 1."}, status=status.HTTP_400_BAD_REQUEST)

            # Concatenar as 5 matrizes 10x10 em um único array achatado (5x100)
            input_data = np.array(data).reshape(5, -1)  # Forma (5, 100)
            
            # Verificar se os labels são válidos
            if len(labels) != 5 or len(set(labels)) != len(labels):
                return Response({"error": "Devem ser fornecidos 5 labels únicos."}, status=status.HTTP_400_BAD_REQUEST)

            # Mapear as letras para valores numéricos (0 a 4)
            label_mapping = {letter: index for index, letter in enumerate(set(labels))}
            output_label = np.array([label_mapping[label] for label in labels])  # Saída numérica correspondente
             # Salvar o mapeamento em um arquivo ou em uma variável global
            with open('letter_mapping.json', 'w') as f:
                json.dump(label_mapping, f)

            # Treinar o MLP com os dados recebidos
            mlp.train(input_data, output_label, epochs=1000, learning_rate=0.1)

            return Response({"message": "Treinamento realizado com sucesso!"}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class MLPredictAPIView(APIView):
    def post(self, request, *args, **kwargs):
        try:
            # Recebendo a matriz de teste
            data = request.data.get('matrix')

            # Verifica a matriz
            if data is None:
                return Response({"error": "Matriz não foi enviada."}, status=status.HTTP_400_BAD_REQUEST)

            # Validar que a matriz é 10x10 com valores 0 ou 1
            if len(data) != 10 or any(len(row) != 10 or any(cell not in [0, 1] for cell in row) for row in data):
                return Response({"error": "A matriz de entrada deve ser 10x10 com valores 0 ou 1."}, status=status.HTTP_400_BAD_REQUEST)

            # Achatar a matriz 10x10 pra virar array
            input_data = np.array(data).reshape(1, -1)  # Forma (1, 100)

            # Carregando o mapa de letras
            with open('letter_mapping.json', 'r') as f:
                label_mapping = json.load(f)

            # Fazendo a previsão usando MLP
            predicted_index = mlp.predict(input_data)  # Isso retorna o índice da previsão

            # Trocando o indice para a letra
            predicted_letter = label_mapping.get(str(predicted_index[0]), "Letra não encontrada")
            
            print(predicted_letter)

            return Response({"predicted_letter": predicted_letter}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

