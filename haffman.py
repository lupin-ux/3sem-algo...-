import heapq
import os
import struct
import json
from collections import defaultdict
from typing import Dict, Tuple, Optional


class HuffmanNode:
    def __init__(self, char: Optional[str] = None, freq: int = 0,
                 left: Optional['HuffmanNode'] = None,
                 right: Optional['HuffmanNode'] = None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other: 'HuffmanNode') -> bool:
        return self.freq < other.freq


class HuffmanCoder:
    def __init__(self):
        self._codebook: Dict[str, str] = {}
        self._reverse_codebook: Dict[str, str] = {}
        self._tree: Optional[HuffmanNode] = None

    def _build_frequency_table(self, data: str) -> Dict[str, int]:
        """Création d'une table de fréquence de symboles"""
        frequency = defaultdict(int)
        for char in data:
            frequency[char] += 1
        return frequency

    def _build_huffman_tree(self, frequency: Dict[str, int]) -> HuffmanNode:
        """Construire un arbre Huffman"""
        heap = []
        for char, freq in frequency.items():
            heapq.heappush(heap, HuffmanNode(char=char, freq=freq))

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, merged)

        return heapq.heappop(heap)

    def _build_codebook(self, node: HuffmanNode, current_code: str = "") -> None:
        """Génération récursive du livre de codes"""
        if node.char is not None:
            self._codebook[node.char] = current_code
            self._reverse_codebook[current_code] = node.char
            return

        self._build_codebook(node.left, current_code + "0")
        self._build_codebook(node.right, current_code + "1")

    def encode(self, data: str) -> Tuple[str, Dict[str, str]]:
        """Codage des données"""
        if not data:
            return "", {}

        frequency = self._build_frequency_table(data)
        self._tree = self._build_huffman_tree(frequency)
        self._codebook = {}
        self._reverse_codebook = {}
        self._build_codebook(self._tree)

        encoded_data = "".join([self._codebook[char] for char in data])
        return encoded_data, self._codebook

    def decode(self, encoded_data: str, codebook: Dict[str, str]) -> str:
        """decode des donnees"""
        if not encoded_data:
            return ""

        # Construire un livre de code inverse pour le décodage
        reverse_codebook = {v: k for k, v in codebook.items()}

        current_code = ""
        decoded_data = []

        for bit in encoded_data:
            current_code += bit
            if current_code in reverse_codebook:
                decoded_data.append(reverse_codebook[current_code])
                current_code = ""

        return "".join(decoded_data)


def read_file(filename: str) -> str:
    """lecture du fichier"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"fichier {filename} inexistant!")

    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


def write_file(filename: str, content: str) -> None:
    """ecriture dans le fichier"""
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)

'''
def save_compressed(encoded_data: str, codebook: Dict[str, str], output_file: str) -> None:
"""Enregistrer les données compressées et le livre de codes"""
# Construire un livre de code inverse pour le décodage
    padding = 8 - len(encoded_data) % 8
    encoded_data += '0' * padding

    bytes_data = bytearray()
    for i in range(0, len(encoded_data), 8):
        byte = encoded_data[i:i + 8]
        bytes_data.append(int(byte, 2))
# Conserver les métadonnées (codebook et padding)
    metadata = {
        'codebook': codebook,
        'padding': padding
    }

    with open(output_file, 'wb') as file:
        file.write(json.dumps(metadata).encode('utf-8') + b'\n')
        file.write(bytes_data)


def load_compressed(input_file: str) -> Tuple[str, Dict[str, str]]:
    """Téléchargement de données compressées"""
    with open(input_file, 'rb') as file:
        metadata_line = file.readline()
        metadata = json.loads(metadata_line.decode('utf-8'))

        codebook = metadata['codebook']
        padding = metadata['padding']

        bytes_data = file.read()
        encoded_data = ''.join([f'{byte:08b}' for byte in bytes_data])

        if padding > 0:
            encoded_data = encoded_data[:-padding]

        return encoded_data, codebook
'''

# === Nouvelle version optimisée ===
def save_compressed(output_file, compressed_data, freq, original_size, compression_time):
    """Version optimisée avec stockage binaire des fréquences"""
    # Pack metadata
    metadata = struct.pack('>QIf', 
                         compressed_data.bit_length(),  # 8 bytes for bits length
                         original_size,                # 4 bytes
                         compression_time)             # 4 bytes
    
    # Pack frequencies
    chars = sorted(freq.keys())
    freq_header = struct.pack('>H', len(chars))  # 2 bytes
    
    freq_body = bytearray()
    for char in chars:
        char_bytes = char.encode('utf-8')
        freq_body.extend([
            len(char_bytes),        # 1 byte
            *char_bytes,           # 1-4 bytes
            *struct.pack('>H', freq[char])  # 2 bytes
        ])
    
    # Write to file
    with open(output_file, 'wb') as f:
        f.write(metadata)          # 16 bytes
        f.write(freq_header)       # 2 bytes
        f.write(freq_body)         # Variable length
        f.write(compressed_data.to_bytes(
            (compressed_data.bit_length() + 7) // 8, 'big'))

def load_compressed(input_file):
    """Version optimisée de chargement"""
    with open(input_file, 'rb') as f:
        # Read metadata (16 bytes)
        metadata = f.read(16)
        bit_length, original_size, compression_time = struct.unpack('>QIf', metadata)
        
        # Read frequencies
        (num_chars,) = struct.unpack('>H', f.read(2))
        
        freq = {}
        for _ in range(num_chars):
            char_len = ord(f.read(1))
            char = f.read(char_len).decode('utf-8')
            freq[char] = struct.unpack('>H', f.read(2))[0]
        
        # Read compressed data
        byte_length = (bit_length + 7) // 8
        compressed_data = int.from_bytes(f.read(byte_length), 'big')
    
    return compressed_data, freq, original_size, compression_time



def compress_file(input_file: str, output_file: str) -> None:
    """compression du fichier"""
    data = read_file(input_file)
    coder = HuffmanCoder()
    encoded_data, codebook = coder.encode(data)

    # conservons les donnees compresses
    save_compressed(encoded_data, codebook, output_file)

    # verification du taux de compression
    original_size = os.path.getsize(input_file)
    compressed_size = os.path.getsize(output_file)
    print(f"taille d_entree: {original_size} байт")
    print(f"taille compressee: {compressed_size} байт")
    print(f"Taux de compression: {original_size / compressed_size:.2f}x")


def decompress_file(input_file: str, output_file: str) -> None:
"""Décompresser le fichier"""
    
    encoded_data, codebook = load_compressed(input_file)
    coder = HuffmanCoder()
    decoded_data = coder.decode(encoded_data, codebook)

    # Enregistrer les données décompressées
    write_file(output_file, decoded_data)

   # Vérifier l'exactitude
    print("Декодирование завершено. Проверьте файл:", output_file)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Huffman coding and decoding')
    parser.add_argument('mode', choices=['encode', 'decode'], help='Режим работы: encode или decode')
    parser.add_argument('input_file', help='Входной файл')
    parser.add_argument('output_file', help='Выходной файл')

    args = parser.parse_args()

    if args.mode == 'encode':
        compress_file(args.input_file, args.output_file)
    else:
        decompress_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
