import os
import pydicom
import numpy as np
import nibabel as nib
from scipy.spatial.distance import euclidean
from Classification3D.utils import KAGGLE_PATH, OUTPUT_PATH
from skimage.transform import resize

def extract_sax_coordinates(base_dir=KAGGLE_PATH+'train/train', output_file=OUTPUT_PATH+'DicomCoordinates.txt'):
    """
    Percorre diretórios de pacientes e extrai as coordenadas (0020,0032) DS Image Position (Patient)
    de arquivos DICOM nas pastas 'sax_', salvando em um arquivo TXT.
    
    :param base_dir: Diretório base que contém os dados dos pacientes.
    :param output_file: Arquivo TXT onde os dados serão salvos.
    """
    data = []

    # Percorre todos os pacientes no diretório base
    for patient_id in os.listdir(base_dir):
        patient_path = os.path.join(base_dir, patient_id)

        # Ignorar arquivos ou itens que não são diretórios
        if not os.path.isdir(patient_path):
            continue
        
        # Verifica se a subpasta 'study' existe
        study_path = os.path.join(patient_path, 'study')
        if not os.path.isdir(study_path):
            print(f"Atenção: A subpasta 'study' não encontrada em {patient_path}.")
            continue
        
        # Percorre as pastas "sax_" dentro da subpasta 'study'
        for folder in os.listdir(study_path):
            if folder.startswith("sax_"):
                sax_path = os.path.join(study_path, folder)
                
                # Certifica-se de que é um diretório
                if not os.path.isdir(sax_path):
                    continue
                
                # Extrair o número da pasta "sax_" para ordenação
                try:
                    sax_number = int(folder.split("_")[1])
                except (IndexError, ValueError):
                    sax_number = float("inf")  # Caso o nome esteja irregular, move para o fim
                
                # Lê todos os arquivos DICOM dentro da pasta
                for dicom_file in os.listdir(sax_path):
                    dicom_path = os.path.join(sax_path, dicom_file)
                    
                    try:
                        # Carrega o arquivo DICOM
                        dicom_data = pydicom.dcmread(dicom_path)

                        # Verifica se o campo (0020,0032) DS Image Position (Patient) existe
                        if (0x0020, 0x0032) in dicom_data:
                            position = dicom_data[(0x0020, 0x0032)].value
                            position_str = ", ".join(map(str, position))
                            
                            # Adiciona as informações na lista de dados
                            data.append((patient_id, sax_number, folder, position_str))
                            break  # Apenas pega uma instância para evitar redundância
                    except Exception as e:
                        # Ignora arquivos inválidos ou erros de leitura
                        print(f"Erro ao ler o arquivo {dicom_path}: {e}")
                        continue

    # Ordena os dados por paciente e número da pasta "sax_"
    data.sort(key=lambda x: (x[0], x[1]))

    # Salva os dados ordenados no arquivo de saída
    with open(output_file, 'w') as out_file:
        out_file.write("Paciente | Pasta SAX | Coordenadas (x, y, z)\n")
        out_file.write("=" * 50 + "\n")
        for entry in data:
            patient_id, sax_number, folder, position_str = entry
            out_file.write(f"{patient_id} | {folder} | {position_str}\n")
    
    print(f"Dados ordenados e salvos em: {output_file}")


def extract_and_select_slices(base_dir=KAGGLE_PATH+'train/train', output_file=OUTPUT_PATH+'SelectedSlices_OrderedByPosition.badListed.txt'):
    """
    Extrai as coordenadas das slices nas pastas 'sax_', ordena-as por posição espacial (z), detecta continuidade
    e cria um arquivo indicando quais slices serão inseridas ou excluídas do volume 3D.
    
    :param base_dir: Diretório base que contém os dados dos pacientes.
    :param output_file: Arquivo TXT onde os resultados serão salvos.
    """

    badList = [("sax_10", "484"), ("sax_7", "282"), ("sax_8", "282"), ("sax_9", "282"), ("sax_10", "11"),
               ("sax_15", "436"), ("sax_8", "436"), ("sax_7", "436"), ("sax_36", "436"), ("sax_21", "282"),
               ("sax_23", "241"), ("sax_90", "195"), ("sax_77", "195"), ("sax_92", "195"), ("sax_80", "195"),
               ("sax_20", "232"), ("sax_8", "393"), ("sax_7", "416"), ("sax_37", "466"), ("sax_16", "280"),
               ("sax_17", "280"), ("sax_18", "280"), ("sax_20", "442"), ("sax_21", "442"), ("sax_22", "442"),
               ("sax_23", "442"), ("sax_24", "442"), ("sax_65", "274"), ("sax_66", "274"), ("sax_67", "274"),
               ("sax_5", "409"), ("sax_6", "409"), ("sax_35", "41"), ("sax_3", "41")]

    with open(output_file, 'w') as out_file:
        out_file.write("Paciente | SAX Folder | Coordenadas (x, y, z) | Distância | Status\n")
        out_file.write("=" * 90 + "\n")
        
        # Percorre todos os pacientes no diretório base
        for patient_id in os.listdir(base_dir):
            patient_path = os.path.join(base_dir, patient_id)

            # Ignorar arquivos ou itens que não são diretórios
            if not os.path.isdir(patient_path):
                continue

            # Verifica se a subpasta 'study' existe
            study_path = os.path.join(patient_path, 'study')
            if not os.path.isdir(study_path):
                print(f"Atenção: A subpasta 'study' não encontrada em {patient_path}.")
                continue

            # Armazenar posições, pastas e coordenadas
            positions = []
            sax_numbers = []
            folders = []

            # Percorre as pastas "sax_" dentro da subpasta 'study'
            for folder in os.listdir(study_path):
                if folder.startswith("sax_"):
                    sax_path = os.path.join(study_path, folder)

                    # Certifica-se de que é um diretório
                    if not os.path.isdir(sax_path):
                        continue

                    # Extrair o número da pasta "sax_" para ordenação
                    try:
                        sax_number = int(folder.split("_")[1])
                    except (IndexError, ValueError):
                        sax_number = float("inf")  # Caso o nome esteja irregular, move para o fim

                    # Lê todos os arquivos DICOM dentro da pasta
                    for dicom_file in os.listdir(sax_path):
                        dicom_path = os.path.join(sax_path, dicom_file)

                        try:
                            # Carrega o arquivo DICOM
                            dicom_data = pydicom.dcmread(dicom_path)

                            # Verifica se o campo (0020,0032) DS Image Position (Patient) existe
                            if (0x0020,0x0032) in dicom_data:
                                position = dicom_data[(0x0020,0x0032)].value
                                frame_number = f"frame_{dicom_data.InstanceNumber}"  # Exemplo para obter frame

                                if (folder, patient_id) in badList:
                                    print(f"Ignorado: {folder}, {patient_id}")
                                    continue

                                # Armazena posição (z), número da pasta e nome para ordenação por posição
                                positions.append((position[2], position))  # Usa apenas o eixo z para ordenar
                                sax_numbers.append(sax_number)
                                folders.append(folder)
                                break  # Apenas pega uma instância para evitar redundância
                        except Exception as e:
                            print(f"Erro ao ler o arquivo {dicom_path}: {e}")
                            continue

            # Ordena as slices com base no eixo z da posição espacial
            sorted_data = sorted(zip(positions, sax_numbers, folders), key=lambda x: x[0][0])  # Ordena pelo eixo z
            positions, sax_numbers, folders = zip(*[(data[0][1], data[1], data[2]) for data in sorted_data])

            valid_positions = []
            valid_folders = []

            if len(positions) > 1:
                mean_distance = np.mean([euclidean(positions[i], positions[i - 1]) for i in range(1, len(positions))])

                # Processa todas as slices
                for i in range(len(positions)):
                    if i == 0 or not valid_positions:
                        # Para a primeira slice, verifica se há uma próxima válida
                        reference_slice = positions[i+1] if i + 1 < len(positions) else None
                    else:
                        # Para demais slices, compara com a última válida
                        reference_slice = valid_positions[-1] if valid_positions else None

                    if reference_slice is not None:
                        distance = euclidean(positions[i], reference_slice)            

                    # Determina continuidade
                    if mean_distance * 0.6 <= distance <= mean_distance * 1.4 and mean_distance < 20:
                        # Slice válida
                        out_file.write(f"{patient_id} | {folders[i]} | {', '.join(map(str, positions[i]))} | {distance:.2f} | Inserida\n")
                        valid_positions.append(positions[i])
                        valid_folders.append(folders[i])
                    else:
                        # Slice inválida
                        out_file.write(f"{patient_id} | {folders[i]} | {', '.join(map(str, positions[i]))} | {distance:.2f} | Excluída (Fora de continuidade)\n")

    print(f"Arquivo com seleção de slices salvo em: {output_file}")

def extract_wrong_coordinates(base_dir=KAGGLE_PATH+'train/train', output_file=OUTPUT_PATH+'WrongCoordinates.txt'):
    """
    Compara coordenadas de todas as slices internas de cada pasta e registra apenas os casos inconsistentes.
    
    :param base_dir: Diretório base que contém os dados dos pacientes.
    :param output_file: Arquivo TXT onde os resultados inconsistentes serão salvos.
    """
    with open(output_file, 'w') as out_file:
        out_file.write("Paciente | SAX Folder | Arquivo | Coordenadas Recebidas | Coordenadas Esperadas\n")
        out_file.write("=" * 100 + "\n")
        
        # Percorre todos os pacientes no diretório base
        for patient_id in os.listdir(base_dir):
            patient_path = os.path.join(base_dir, patient_id)

            if not os.path.isdir(patient_path):
                continue

            study_path = os.path.join(patient_path, 'study')
            if not os.path.isdir(study_path):
                continue

            # Percorre as pastas "sax_" dentro da subpasta 'study'
            for folder in os.listdir(study_path):
                if folder.startswith("sax_"):
                    sax_path = os.path.join(study_path, folder)

                    if not os.path.isdir(sax_path):
                        continue

                    # Armazena coordenadas para comparação
                    positions = []
                    file_names = []

                    for dicom_file in os.listdir(sax_path):
                        dicom_path = os.path.join(sax_path, dicom_file)

                        try:
                            # Carrega o arquivo DICOM
                            dicom_data = pydicom.dcmread(dicom_path)

                            # Verifica se o campo (0020,0032) DS Image Position (Patient) existe
                            if (0x0020, 0x0032) in dicom_data:
                                position = dicom_data[(0x0020, 0x0032)].value
                                positions.append(position)  # Adiciona coordenadas
                                file_names.append(dicom_file)  # Adiciona o nome do arquivo
                        except Exception as e:
                            print(f"Erro ao ler o arquivo {dicom_path}: {e}")
                            continue
                    
                    # Verifica consistência de coordenadas dentro da pasta
                    if len(positions) > 1:
                        expected_position = positions[0]  # Considera a primeira como esperada
                        for i, position in enumerate(positions):
                            if not np.allclose(position, expected_position, atol=1e-3):
                                # Caso inconsistente: Escreve no arquivo
                                out_file.write(f"{patient_id} | {folder} | {file_names[i]} | {position} | {expected_position}\n")
    
    print(f"Casos inconsistentes foram salvos no arquivo: {output_file}")

def load_dicom_series(base_dir=KAGGLE_PATH+'train/train'):
    slices = []
    positions = []
    for patient_id in os.listdir(base_dir):
        patient_path = os.path.join(base_dir, patient_id)

        # Ignorar arquivos ou itens que não são diretórios
        if not os.path.isdir(patient_path):
            continue
        
        # Verifica se a subpasta 'study' existe
        study_path = os.path.join(patient_path, 'study')
        if not os.path.isdir(study_path):
            print(f"Atenção: A subpasta 'study' não encontrada em {patient_path}.")
            continue
        
        # Percorre as pastas "sax_" dentro da subpasta 'study'
        for folder in os.listdir(study_path):
            if folder.startswith("sax_"):
                sax_path = os.path.join(study_path, folder)
                
                # Certifica-se de que é um diretório
                if not os.path.isdir(sax_path):
                    continue
                
                # Extrair o número da pasta "sax_" para ordenação
                try:
                    sax_number = int(folder.split("_")[1])
                except (IndexError, ValueError):
                    sax_number = float("inf")  # Caso o nome esteja irregular, move para o fim
                
                # Lê todos os arquivos DICOM dentro da pasta
                for dicom_file in os.listdir(sax_path):
                    try:
                        dicom_data = pydicom.dcmread(os.path.join(sax_path, dicom_file))
                        slices.append(dicom_data.pixel_array)
                        positions.append((dicom_data.ImagePositionPatient, dicom_data.SliceLocation))
                    except Exception as e:
                            print(f"Erro ao ler o arquivo {sax_path}: {e}")
                            continue
                    
    return slices, positions

def load_dicom_series(base_dir, output_path=OUTPUT_PATH, output_filename="1_4d.nii"):
    """
    Processa uma série de pastas contendo arquivos DICOM para criar um volume NIfTI 4D (3D + tempo).

    :param base_dir: Diretório base com pastas que contêm arquivos DICOM.
    :param output_path: Diretório onde o volume NIfTI será salvo.
    :param output_filename: Nome do arquivo de saída (default: 1_4d.nii).
    """
    all_volumes = []  # Lista para armazenar volumes 3D individuais
    reference_shape = None  # Dimensão de referência para redimensionar slices

    # Itera pelas pastas externas
    for folder in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):  # Certifica-se de que é uma pasta
            continue

        slices = []
        positions = []

        # Itera pelos arquivos DICOM dentro de cada pasta
        for file in sorted(os.listdir(folder_path), key=lambda x: int(x.split('-')[-1].split('.')[0])):
            if file.endswith(".dcm"):
                dicom_path = os.path.join(folder_path, file)
                dicom_data = pydicom.dcmread(dicom_path)

                if hasattr(dicom_data, "ImagePositionPatient"):
                    pixel_array = dicom_data.pixel_array
                    position = np.array(dicom_data.ImagePositionPatient, dtype=np.float32)

                    # Define a dimensão de referência na primeira fatia
                    if reference_shape is None:
                        reference_shape = pixel_array.shape

                    # Redimensiona fatias para a dimensão de referência
                    if pixel_array.shape != reference_shape:
                        print(f"Ajustando dimensão de {file} de {pixel_array.shape} para {reference_shape}")
                        pixel_array = resize(pixel_array, reference_shape, mode='constant', preserve_range=True).astype(np.uint16)

                    slices.append(pixel_array)
                    positions.append(position)

        if slices:
            # Determina a coordenada média z da pasta
            mean_position_z = np.mean([pos[2] for pos in positions])

            # Empilha as fatias da pasta atual em um volume 3D
            volume_3d = np.stack(slices, axis=0)  # (número de fatias, altura, largura)
            all_volumes.append((volume_3d, mean_position_z))

    # Ordena os volumes 3D com base no eixo z
    all_volumes_sorted = sorted(all_volumes, key=lambda x: x[1])  # Ordena pelo valor z
    sorted_volumes = [vol[0] for vol in all_volumes_sorted]  # Apenas os volumes, já ordenados

    if sorted_volumes:
        # Empilha os volumes 3D para criar um volume 4D
        volume_4d = np.stack(sorted_volumes, axis=-1)  # Adiciona a dimensão de tempo/espaço
        volume_4d = np.transpose(volume_4d, [1, 2, 3, 0])  # Ajusta para [altura, largura, tempo, outras]

        # Matriz affine (ajustando a profundidade)
        affine = np.eye(4)
        scale_factor = 256 / volume_4d.shape[2]  # Escala para "esticar" o eixo de profundidade
        affine[2, 2] = scale_factor  # Ajusta o tamanho do voxel no eixo z (profundidade)

        # Salva o volume NIfTI
        nib.save(nib.Nifti1Image(volume_4d, affine), os.path.join(output_path, output_filename))
        print(f"Volume 4D salvo em: {os.path.join(output_path, output_filename)}")
        print(f"Shape final do volume 4D: {volume_4d.shape}")
    else:
        print("Nenhuma pasta DICOM válida foi processada.")
    print(volume_4d.shape)

# stack, positions = load_dicom_series()
# Executa o pipeline de seleção e grava os resultados no arquivo
# extract_and_select_slices()

# Executa a extração e salvamento
# extract_sax_coordinates()

# extract_wrong_coordinates()
load_dicom_series(KAGGLE_PATH+'train/train/1/study')

