from groq import Groq
from tech_doc_rag.config.settings import settings
from tech_doc_rag.generation.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


class GroqLLMService:
    """Groq API üzerinden LLM yanıt üretimi sağlayan servis."""

    def __init__(self):
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY ortam değişkeni bulunamadı! .env dosyasını kontrol edin.")
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL

    def generate_response(self, query: str, context_text: str) -> str:
        """Getirilen context metni ve sorgu ile LLM'den yanıt üretir."""
        user_content = USER_PROMPT_TEMPLATE.format(
            context_text=context_text,
            query=query
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.1,  # Tutarlı ve doğrudan sonuçlar için düşük sıcaklık
            max_tokens=1024,
        )

        return response.choices[0].message.content
