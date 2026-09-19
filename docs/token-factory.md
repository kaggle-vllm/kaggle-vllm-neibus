# Nebius Token Factory

The official Nebius Nemotron notebook uses the OpenAI Python SDK as a protocol
client:

```python
OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ["NEBIUS_API_KEY"],
)
```

That does not call the OpenAI hosted API. The request is sent to Nebius Token
Factory.

Primary project model:

`nvidia/Nemotron-3_5-Lightning`
