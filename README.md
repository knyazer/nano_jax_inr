You should be able to run the code if you do:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/knyazer/nano_jax_inr
cd nano_jax_inr
uv pip install -r pyproject.toml
uv run main.py
```

wait for a bit, it will download a few gigs of things, and then take 30 secs to run or smth
