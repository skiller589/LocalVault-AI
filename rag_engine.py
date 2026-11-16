import os
import io
import json
import requests
import pypdf

class LocalVaultRAG:
    def __init__(self, storage_dir="data"):
        self.storage_dir = storage_dir
        self.model = "llama3:latest"
        self.documents = []  # In-memory document storage for fast local testing
        self.ollama_url = "http://localhost:11434/api/generate"
        os.makedirs(storage_dir, exist_ok=True)

    def set_model(self, model_name: str):
        self.model = model_name

    def extract_text(self, uploaded_file) -> str:
        filename = uploaded_file.name.lower()
        if filename.endswith(".pdf"):
            reader = pypdf.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        else:
            return uploaded_file.read().decode("utf-8", errors="ignore")

    def ingest_files(self, files) -> int:
        count = 0
        for f in files:
            text = self.extract_text(f)
            if text.strip():
                # Simple chunking (500 chars with 100 overlap)
                chunks = self._chunk_text(text, chunk_size=600, overlap=100)
                for chunk in chunks:
                    self.documents.append({
                        "file": f.name,
                        "text": chunk
                    })
                count += 1
        return count

    def _chunk_text(self, text: str, chunk_size=600, overlap=100):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += chunk_size - overlap
        return chunks

    def search_context(self, query: str, top_k=3):
        if not self.documents:
            return []
        
        # Keyword-based relevance score for lightweight on-device retrieval
        words = set(query.lower().split())
        scored = []
        for doc in self.documents:
            doc_words = doc["text"].lower()
            score = sum(1 for w in words if w in doc_words)
            if score > 0:
                scored.append((score, doc))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [item[1] for item in scored[:top_k]]
        
        # If no direct match, return latest docs as fallback context
        if not results and self.documents:
            results = self.documents[:top_k]
            
        return results

    def query(self, prompt: str):
        context_docs = self.search_context(prompt)
        
        context_str = ""
        sources = []
        for doc in context_docs:
            context_str += f"\n---\nExtrait du fichier '{doc['file']}':\n{doc['text']}\n"
            snippet = doc['text'][:120] + "..." if len(doc['text']) > 120 else doc['text']
            sources.append({"file": doc["file"], "snippet": snippet})

        system_instruction = (
            "Tu es LocalVault AI, un assistant personnel local et confidentiel. "
            "Réponds en français de manière claire et concise. "
            "Utilise les extraits de documents suivants si pertinents pour étayer ta réponse :\n"
            f"{context_str}\n\n"
            f"Question de l'utilisateur : {prompt}"
        )

        # Call local Ollama API with fast fallback
        try:
            # Check if Ollama is accessible first with 2s timeout
            requests.get("http://localhost:11434/api/tags", timeout=2)
            
            payload = {
                "model": self.model,
                "prompt": system_instruction,
                "stream": False
            }
            resp = requests.post(self.ollama_url, json=payload, timeout=45)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response", "Réponse vide."), sources
            else:
                return (
                    f"⚠️ Ollama a répondu avec le code {resp.status_code}. "
                    f"Assurez-vous que le modèle '{self.model}' est téléchargé (`ollama pull {self.model}`).",
                    sources
                )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # Fallback demonstration mode if Ollama server is not running
            mock_answer = (
                f"💡 **[Mode Démonstration LocalVault AI]**\n\n"
                f"J'ai bien analysé votre question : *\"{prompt}\"* dans vos fichiers locaux.\n\n"
                f"ℹ️ *Note : Pour exécuter le modèle LLM complet en local avec accélération GPU, "
                f"lancez Ollama en arrière-plan (`ollama run {self.model}`).*\n\n"
                f"**Passages les plus pertinents trouvés dans vos documents :**\n"
            )
            for s in sources:
                mock_answer += f"- **{s['file']}** : « *{s['snippet']}* »\n"
            
            return mock_answer, sources
