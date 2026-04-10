# Quick Start: Setup Modal Secrets

## Option 1: One-Line Command (Fastest)

Run this command in your terminal, replacing `<YOUR_API_KEY>` with your actual Modal API key:

```bash
modal secret create mercury-moe-secrets MODAL_API_KEY=<YOUR_API_KEY>
```

## Option 2: Interactive Mode

```bash
modal secret create mercury-moe-secrets
```

Then follow the prompts to add your `MODAL_API_KEY`.

## Option 3: Via Modal Dashboard

1. Go to [https://modal.com/secrets](https://modal.com/secrets)
2. Click **"Create Secret"**
3. Name it: `mercury-moe-secrets`
4. Add key: `MODAL_API_KEY`
5. Paste your API key value
6. Click **Save**

---

## Verify Setup

After creating the secret, verify it exists:

```bash
modal secret list
```

You should see `mercury-moe-secrets` in the list.

---

## Next Steps

Once your secret is set up:

1. **Upload your data:**
   ```bash
   python upload_data.py --data_dir <your_data_path>
   ```

2. **Run training:**
   ```bash
   modal run modal_app.py::train
   ```

3. **Or get an interactive shell:**
   ```bash
   modal shell modal_app.py
   ```

---

## What We Configured

Your `modal_app.py` now:
- ✅ Loads secrets from `mercury-moe-secrets`
- ✅ Makes `MODAL_API_KEY` available as an environment variable in your Modal functions
- ✅ Gracefully handles cases where the secret doesn't exist yet (won't crash)
- ✅ Uses L4 GPUs with persistent volumes for data/checkpoints/logs

The secret will be automatically injected into all training, evaluation, and shell functions!
