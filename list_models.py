"""Print the models your key can use, so you can pick a name for MODEL."""
from agent_gemini import make_client

client = make_client()  # keep a reference, otherwise the client gets closed
for m in client.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(m.name.replace("models/", ""))