# 🔒 LocalVault AI — Votre PC. Votre Agent.

[![ASUS Challenge](https://img.shields.io/badge/ASUS%20Agent-Computer%20Challenge-00539b?style=for-the-badge)](https://www.asus.com)
[![Privacy](https://img.shields.io/badge/Privacy-100%25%20On--Device-059669?style=for-the-badge)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-gray?style=for-the-badge)](LICENSE)

> **Projet soumis au ASUS Agent Computer Challenge**  
> *L'agent d'assistance documentaire autonome et 100 % hors-ligne pour exploiter la puissance de votre GPU local.*

---

## 💡 Présentation du Projet

**LocalVault AI** est un agent conversationnel et d'organisation documentaire fonctionnant **en circuit fermé sur votre PC**. 

Il permet d'interroger en langage naturel vos documents personnels (cours universitaires, notices techniques, notes de travail, factures) sans jamais envoyer le moindre octet sur le cloud.

### 🌟 Fonctionnalités Clés
- **100% On-Device & Zero-Cloud :** Aucune fuite de données, respect total de la vie privée.
- **RAG Local Instantané :** Indexation de fichiers PDF, TXT et Markdown.
- **Réponses Sourcées :** Chaque réponse cite le document et le passage d'origine pour éliminer les hallucinations.
- **Compatible Modèles Libres :** Pilote Llama 3 (8B), Mistral (7B) ou Phi-3 via Ollama en local.

---

## 🛠️ Architecture Système

```
[ Vos Fichiers Locaux ] ➔ [ Extraction & Chunking ] ➔ [ Indexation RAG ]
                                                                 │
[ Vous (Prompt FR) ] ➔ [ Recherche Sémantique ] ➔ [ LLM Local (Ollama) ] ➔ [ Réponse Sourcée ]
```

---

## 🖥️ Pourquoi l'écosystème matériel ASUS est déterminant ?

Sur un PC portable d'entrée de gamme équipé de **6 Go de VRAM** (ex. RTX 4050 Laptop), l'analyse de documents volumineux sature rapidement la mémoire vidéo, provoquant un délestage lent sur le processeur (CPU) et une chute de vitesse de génération.

**Le bond de performance avec l'upgrade ASUS :**
- **ASUS GeForce RTX 5060 Ti 16GB :** Le triplement de la VRAM (16 Go) permet de loger le modèle LLM et la totalité du contexte documentaire en mémoire vidéo ultra-rapide (> 40 tokens/sec sans latence).
- **Carte Mère ASUS PRIME B860-PLUS :** Stabilité thermique et bande passante PCIe 5.0 pour un chargement instantané des modèles lourds depuis le SSD NVMe vers le GPU.

---

## 🚀 Démarrage Rapide

### 1. Prérequis
- Python 3.10 ou supérieur
- [Ollama](https://ollama.ai) installé pour l'inférence locale

### 2. Installation
```bash
# Cloner le dépôt
git clone https://github.com/<votre-nom-utilisateur>/LocalVault-AI.git
cd LocalVault-AI

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Télécharger le modèle local
```bash
# Dans un terminal, lancer le modèle Llama 3 (8B)
ollama run llama3
```

### 4. Lancer l'Agent LocalVault AI
```bash
streamlit run app.py
```
L'interface s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`.

---

## 📄 Licence
Ce projet est sous licence MIT — vous êtes libre de l'utiliser et de le modifier.
