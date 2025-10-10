# pipeline/transformer.py
class Transformer:
    def clean_emails(self, users):
        print("🔹 Cleaning email addresses...")
        cleaned = []
        for user in users:
            user_id, name, email = user
            email = email.lower().strip()
            cleaned.append((user_id, name, email))
        print("✅ Emails cleaned")
        return cleaned
