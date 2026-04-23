#!/usr/bin/env python3
"""
Direct diagnostic for criticism_lora — tests it with a full-length pipeline prompt.
"""
import httpx

planner_out = """Here is a plan for designing and implementing a JWT-based authentication microservice:

1. Define requirements:
   - User registration (POST /register): accepts username, password
   - User login (POST /login): returns JWT access token + refresh token
   - Token validation (GET /validate): verifies and decodes JWT
   - Token refresh (POST /refresh): rotates refresh token
   - Logout (POST /logout): blacklists the refresh token
   - Role-based access control (RBAC) for admin/user roles

2. Choose the technology stack:
   - Framework: FastAPI (async, modern, OpenAPI auto-docs)
   - Database: PostgreSQL via asyncpg/SQLAlchemy async
   - Token library: python-jose[cryptography] or PyJWT
   - Password hashing: passlib with bcrypt backend
   - Token blacklist: Redis with TTL matching token expiry

3. Design the data models:
   - User(id UUID, username str, password_hash str, role Enum, created_at datetime)
   - RefreshToken(id UUID, user_id UUID FK, token_hash str, expires_at datetime)

4. Define the security architecture:
   - RS256 asymmetric signing: private key signs tokens, public key validates
   - Access token TTL: 15 minutes
   - Refresh token TTL: 7 days
   - Rate limiting: slowapi (token bucket, 10 req/min per IP)
   - HTTPS-only in production

5. Implement the microservice:
   - FastAPI app with Pydantic v2 settings
   - Async SQLAlchemy engine + Alembic migrations
   - JWT utils: generate_access_token(), decode_token()
   - Password utils: hash_password(), verify_password()
   - Endpoints with dependency injection

6. Testing and Deployment:
   - Unit tests + integration tests with pytest-asyncio
   - Docker multi-stage build, docker-compose for local dev
   - K8s manifests for production"""

researcher_out = """Research Summary: JWT Authentication Best Practices

Key Findings:
JWT (JSON Web Token) authentication is a stateless approach (RFC 7519) widely adopted for microservice auth. Tokens encode claims signed with HMAC-SHA256 or RSA.

Architecture Best Practices:
1. Token Separation: Issue short-lived access tokens (5-15 min) and long-lived refresh tokens (7-30 days).
2. Asymmetric Signing: RS256 is preferred. Auth service holds private key; other services validate with JWKS.
3. JWKS Endpoint: Expose GET /.well-known/jwks.json for public key rotation.
4. Token Revocation: Use Redis sorted sets with TTL for denylist (O(1) lookup, auto-expiry).
5. Refresh Token Rotation: Rotate-on-use pattern prevents replay attacks.

Security Considerations (OWASP API Security Top 10):
- API2: Use RS256 not HS256 in distributed settings
- API3: Never return password_hash in responses
- API8: Set HttpOnly, Secure, SameSite=Strict cookie flags
- Validate alg field: reject alg:none tokens

Recommended Stack: FastAPI + python-jose + passlib + asyncpg + Redis"""

url = "http://localhost:8001/v1/completions"
prompt = (
    "[SYSTEM]\n"
    "You are CriticAgent - a ruthless quality gatekeeper.\n"
    "Your job: give an honest overall assessment of the full solution so far.\n"
    "Output:\n"
    "SCORE: <0.0-1.0>\n"
    "VERDICT: <one sentence summary>\n"
    "BLOCKERS: <critical issues that must be fixed, or 'None'>\n"
    "IMPROVEMENT: <the single most impactful change>\n\n"
    "[USER]\n"
    "Prior agent outputs:\n"
    f"[iter 1 -- planner]: {planner_out}\n\n"
    f"[iter 1 -- researcher]: {researcher_out}\n\n"
    "[ASSISTANT]\n"
    "I've reviewed the prior outputs.\n\n"
    "[USER]\n"
    "Plan the design and implementation roadmap for a JWT-based authentication microservice in Python.\n\n"
    "[RESPONSE]\n"
)
print(f"Prompt length: {len(prompt)} chars, ~{len(prompt)//3} estimated tokens")

payload = {
    "model": "criticism_lora",
    "prompt": prompt,
    "max_tokens": 384,
    "temperature": 0.2,
    "stop": ["[SYSTEM]", "[USER]", "[ASSISTANT]", "\n[RESPONSE]"],
}
r = httpx.post(url, json=payload, timeout=90)
data = r.json()
c = data["choices"][0]
print(f"FINISH: {c['finish_reason']}")
print(f"LEN:    {len(c['text'])}")
print(f"TEXT:   {repr(c['text'][:400])}")
print()

# Also test WITHOUT the \n[RESPONSE] stop token to see what happens
print("--- Without \\n[RESPONSE] stop token ---")
payload2 = dict(payload)
payload2["stop"] = ["[SYSTEM]", "[USER]", "[ASSISTANT]"]
r2 = httpx.post(url, json=payload2, timeout=90)
data2 = r2.json()
c2 = data2["choices"][0]
print(f"FINISH: {c2['finish_reason']}")
print(f"LEN:    {len(c2['text'])}")
print(f"TEXT:   {repr(c2['text'][:400])}")
