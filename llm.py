"""LLM providers, resume compression, job-matching prompts, and analysis pipeline."""

import hashlib
import json
import math
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from docx import Document  # type: ignore

from log_util import debug as log_debug
from log_util import info, verbose

DEFAULT_JOBS_PER_LLM_BATCH = 30
MAX_LLM_JOBS_PER_ROUND = 12

__all__ = [
    "DEFAULT_JOBS_PER_LLM_BATCH",
    "MAX_LLM_JOBS_PER_ROUND",
    "LLMProvider",
    "GeminiProvider",
    "ClaudeProvider",
    "ChatGPTProvider",
    "create_llm_provider",
    "get_resume_for_matching",
    "analyze_jobs",
    "analyze_jobs_batched",
    "merge_scores_with_jobs",
    "parse_llm_job_scores",
    "normalize_scored_jobs",
    "clean_duplicate_output",
    "validate_results",
]

# =============================================================================
# LLM Provider Abstraction
# =============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, provider_config: dict[str, Any]):
        """Initialize the provider with its configuration.
        
        Args:
            provider_config: Provider-specific configuration dict (api_key, model, etc.)
        """
        self.config = provider_config
    
    def _resolve_api_key(self) -> str:
        """Resolve the API key from config or environment variable.
        
        Priority:
        1. Environment variable (if api_key_env_var is set and the env var exists)
        2. Direct api_key from config
        
        Returns:
            The resolved API key.
            
        Raises:
            ValueError: If no API key can be resolved.
        """
        # Check for environment variable override first
        env_var_name = self.config.get("api_key_env_var")
        if env_var_name:
            env_key = os.environ.get(env_var_name)
            if env_key:
                return env_key
        
        # Fall back to direct api_key in config
        api_key = self.config.get("api_key")
        if api_key:
            return api_key
        
        # No key found - provide helpful error message
        if env_var_name:
            raise ValueError(
                f"{self.name} provider requires an API key. "
                f"Set the '{env_var_name}' environment variable or provide 'api_key' in configuration."
            )
        else:
            raise ValueError(
                f"{self.name} provider requires 'api_key' in configuration, "
                f"or set 'api_key_env_var' to specify an environment variable name."
            )
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the display name of this provider."""
        pass
    
    @property
    @abstractmethod
    def model(self) -> str:
        """Return the model being used."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM.
            
        Returns:
            The text response from the LLM.
        """
        pass

    @property
    def jobs_per_llm_batch(self) -> int:
        """Target batch size from provider config (keys off llm_providers.<name>.jobs_per_llm_batch)."""
        v = self.config.get("jobs_per_llm_batch")
        if v is not None:
            try:
                n = int(v)
            except (TypeError, ValueError):
                return DEFAULT_JOBS_PER_LLM_BATCH
            return max(1, n)
        return DEFAULT_JOBS_PER_LLM_BATCH


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider."""
    
    def __init__(self, provider_config: dict[str, Any]):
        super().__init__(provider_config)
        # Import here to avoid global dependency
        from google import genai
        
        api_key = self._resolve_api_key()
        self._client = genai.Client(api_key=api_key)
        # Require explicit model selection - no default since this costs money
        self._model = provider_config.get("model")
        if not self._model:
            raise ValueError(
                "Gemini provider requires 'model' to be specified in configuration. "
                "Example models: gemini-2.0-flash (cheap), gemini-2.0-pro"
            )
    
    @property
    def name(self) -> str:
        return "Gemini"
    
    @property
    def model(self) -> str:
        return self._model
    
    def generate(self, prompt: str) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s
                        verbose(f"  Rate limited. Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise
        return ""


class ClaudeProvider(LLMProvider):
    """Anthropic Claude LLM provider."""
    
    def __init__(self, provider_config: dict[str, Any]):
        super().__init__(provider_config)
        # Import here to avoid global dependency
        import anthropic
        
        api_key = self._resolve_api_key()
        self._client = anthropic.Anthropic(api_key=api_key)
        # Require explicit model selection - no default since this costs money
        self._model = provider_config.get("model")
        if not self._model:
            raise ValueError(
                "Claude provider requires 'model' to be specified in configuration. "
                "Example models: claude-sonnet-4-20250514 (balanced), claude-opus-4-20250514 (best)"
            )
        self._max_tokens = provider_config.get("max_tokens", 16384)
        # Store anthropic module for exception handling
        self._anthropic = anthropic
    
    @property
    def name(self) -> str:
        return "Claude"
    
    @property
    def model(self) -> str:
        return self._model
    
    def generate(self, prompt: str) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use streaming to handle large prompts that take >10 minutes
                result_text = ""
                with self._client.messages.stream(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                ) as stream:
                    for text in stream.text_stream:
                        result_text += text
                return result_text
            except self._anthropic.RateLimitError:
                if attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s
                    verbose(f"  Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = 30 * (attempt + 1)
                        verbose(f"  Rate limited. Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise
        return ""


class ChatGPTProvider(LLMProvider):
    """OpenAI ChatGPT LLM provider."""
    
    def __init__(self, provider_config: dict[str, Any]):
        super().__init__(provider_config)
        # Import here to avoid global dependency
        import openai
        
        api_key = self._resolve_api_key()
        self._client = openai.OpenAI(api_key=api_key)
        # Require explicit model selection - no default since this costs money
        self._model = provider_config.get("model")
        if not self._model:
            raise ValueError(
                "ChatGPT provider requires 'model' to be specified in configuration. "
                "Example models: gpt-4o-mini (cheap), gpt-4o, gpt-5.2 (best)"
            )
        self._max_tokens = provider_config.get("max_tokens", 16384)
        # Store openai module for exception handling
        self._openai = openai
    
    @property
    def name(self) -> str:
        return "ChatGPT"
    
    @property
    def model(self) -> str:
        return self._model
    
    def generate(self, prompt: str) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use streaming to handle large prompts
                result_text = ""
                stream = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        result_text += chunk.choices[0].delta.content
                return result_text
            except self._openai.AuthenticationError as e:
                raise ValueError(
                    f"OpenAI authentication failed. Check your API key. Error: {e}"
                ) from e
            except self._openai.PermissionDeniedError as e:
                raise ValueError(
                    f"OpenAI permission denied. Your account may lack billing setup or access to model '{self._model}'. Error: {e}"
                ) from e
            except self._openai.NotFoundError as e:
                raise ValueError(
                    f"OpenAI model '{self._model}' not found. Check the model name. Error: {e}"
                ) from e
            except self._openai.RateLimitError as e:
                # Check if it's actually a quota/billing issue vs true rate limit
                error_str = str(e).lower()
                if "quota" in error_str or "billing" in error_str or "insufficient" in error_str:
                    raise ValueError(
                        f"OpenAI quota exceeded or billing issue. Check your OpenAI account. Error: {e}"
                    ) from e
                if attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s
                    verbose(f"  Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                error_str = str(e).lower()
                # Only retry on actual rate limits, not other errors
                if "429" in str(e) and "rate" in error_str and "quota" not in error_str:
                    if attempt < max_retries - 1:
                        wait_time = 30 * (attempt + 1)
                        verbose(f"  Rate limited. Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise
        return ""


def create_llm_provider(provider_name: str, provider_config: dict[str, Any]) -> LLMProvider:
    """Factory function to create an LLM provider instance.
    
    Args:
        provider_name: Name of the provider ("gemini", "claude", or "chatgpt").
        provider_config: Provider-specific configuration dict.
        
    Returns:
        An initialized LLMProvider instance.
        
    Raises:
        ValueError: If provider_name is not recognized.
    """
    providers = {
        "gemini": GeminiProvider,
        "claude": ClaudeProvider,
        "chatgpt": ChatGPTProvider,
    }
    
    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        valid_providers = ", ".join(providers.keys())
        raise ValueError(f"Unknown LLM provider '{provider_name}'. Valid options: {valid_providers}")
    
    return provider_class(provider_config)
def load_resume_text(resume_path: str | Path) -> str:
    path = Path(resume_path)
    if not path.exists():
        info(f"Resume file '{path}' not found. Continuing with empty resume text.")
        return ""

    if path.suffix.lower() == ".docx":
        try:
            doc = Document(path)
            # Combine paragraphs with blank lines preserved
            return "\n".join(p.text for p in doc.paragraphs if p.text).strip()
        except Exception as exc:  # noqa: BLE001
            info(f"Could not read DOCX resume '{path}': {exc}. Using empty resume text.")
            return ""

    # Fallback to plain-text read
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as exc:  # noqa: BLE001
        info(f"Could not read resume file '{path}': {exc}. Using empty resume text.")
        return ""


# =============================================================================
# Resume Caching and Compression
# =============================================================================

RESUME_CACHE_FILE = Path("resume.cached")

RESUME_COMPRESSION_PROMPT = """You are a resume compressor for job-matching.

INPUT: A resume in plain text or Markdown.
OUTPUT: A compact plain-text "LLM Matching Payload" suitable to compare against job descriptions.

Hard constraints:
- Output must be <= 2400 characters.
- Output must be plain text only (no code fences).
- Do NOT invent skills, tools, or experience not explicitly present in the resume.
- Preserve important duty wording and all technologies mentioned in duties.

What to KEEP (in this exact structure):

NAME | Location
Headline: <short role label>

Skills: <single line, comma-separated; include languages, cloud, infra, databases, core backend keywords found in resume>

Experience
<Most recent company> — <title> (<dates>, <location/remote>)
• <3–6 duty bullets: copy or lightly compress the resume's real duties; MUST retain architecture/scale/reliability details and tech stack>

<Second most recent company> — <title> (<dates>, <location/remote>)
• <3–5 duty bullets: same rules>

Continue through last minimum 10 years of jobs 

Earlier roles (high level)
• <Company> — <Title>: <ONE sentence duty/impact summary copied/condensed from resume>
• <Company> — <Title>: <ONE sentence duty/impact summary copied/condensed from resume>

Education: <one line, degree + school>

What to REMOVE:
- phone, email, LinkedIn, GitHub URLs
- long summaries, soft-skill fluff, repeated headings
- projects unless they contain unique technologies not shown elsewhere
- any duty bullets that are purely generic (e.g., "collaborated with team") unless they include unique tech or measurable impact

Normalization rules:
- Collapse whitespace; keep the bullet marker "•"
- Prefer concrete duties that indicate fit: backend services, infra, reliability, distributed systems, data pipelines, CI/CD, production ops

Now produce the compact payload from the resume below:

"""

RESUME_VERIFICATION_PROMPT = """You are a resume verification assistant.

Your task is to verify that a condensed resume accurately represents the original full resume.

IMPORTANT RULES:
- The condensed version should capture all key skills, technologies, job titles, companies, and dates.
- It should NOT invent or add anything not in the original.
- It should NOT omit critical technical skills, job roles, or significant achievements.
- Minor wording differences are acceptable if the meaning is preserved.
- The condensed version must be <= 2400 characters.

Below is the ORIGINAL FULL RESUME:
---
{full_resume}
---

Below is the CONDENSED VERSION:
---
{condensed_resume}
---

If the condensed version is an accurate representation of the original:
- Reply with EXACTLY: ACCURATE

If the condensed version has issues (missing critical info, invented content, or significant misrepresentation):
- Reply with: NEEDS_CORRECTION
- Then provide a corrected condensed version on the next line (plain text, no code fences, <= 2400 chars)

Your response:
"""


def compute_resume_checksum(resume_text: str) -> str:
    """Compute SHA256 checksum of the resume text.
    
    Args:
        resume_text: The full resume text content.
        
    Returns:
        Hex string of the SHA256 hash.
    """
    return hashlib.sha256(resume_text.encode("utf-8")).hexdigest()


def load_resume_cache(cache_path: Path = RESUME_CACHE_FILE) -> dict | None:
    """Load the cached compressed resume if it exists.
    
    Args:
        cache_path: Path to the cache file.
        
    Returns:
        Dict with 'checksum', 'payload', 'timestamp' or None if not found.
    """
    if not cache_path.exists():
        return None
    
    try:
        cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
        # Validate required fields
        if all(k in cache_data for k in ("checksum", "payload", "timestamp")):
            return cache_data
    except (json.JSONDecodeError, OSError) as e:
        info(f"  Warning: Could not read resume cache: {e}")
    
    return None


def save_resume_cache(
    checksum: str, 
    payload: str, 
    cache_path: Path = RESUME_CACHE_FILE
) -> None:
    """Save the compressed resume to cache.
    
    Args:
        checksum: SHA256 checksum of the original resume.
        payload: The compressed resume text.
        cache_path: Path to the cache file.
    """
    cache_data = {
        "checksum": checksum,
        "payload": payload,
        "timestamp": datetime.now().isoformat(),
    }
    cache_path.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
    log_debug(f"  Resume cache saved to {cache_path}")


def compress_resume(resume_text: str, provider: "LLMProvider") -> str:
    """Compress the resume using the LLM for job-matching.
    
    This is a two-stage process:
    1. Initial compression to create a condensed version
    2. Verification that the condensed version accurately represents the original
    
    Args:
        resume_text: The full resume text.
        provider: The LLM provider to use.
        
    Returns:
        Compressed resume payload (<= 2400 chars).
    """
    verbose(f"  Stage 1: Compressing resume using {provider.name} ({provider.model})...")
    
    prompt = RESUME_COMPRESSION_PROMPT + resume_text
    compressed = provider.generate(prompt)
    
    # Strip any code fences the LLM might have added
    compressed = _strip_code_fences(compressed)
    
    # Warn if over limit
    if len(compressed) > 2400:
        info(f"  Warning: Initial compressed resume is {len(compressed)} chars (target: 2400)")
    else:
        log_debug(f"  Initial compressed resume: {len(compressed)} chars")
    
    # Stage 2: Verify the compressed resume
    compressed = verify_compressed_resume(resume_text, compressed, provider)
    
    return compressed


def _strip_code_fences(text: str) -> str:
    """Strip code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if "```" in text:
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    return text


def verify_compressed_resume(
    full_resume: str, 
    compressed_resume: str, 
    provider: "LLMProvider"
) -> str:
    """Verify and potentially correct the compressed resume.
    
    Stage 2 of the compression process: asks the LLM to verify that the 
    condensed version accurately represents the original full resume.
    
    Args:
        full_resume: The original full resume text.
        compressed_resume: The initially compressed resume.
        provider: The LLM provider to use.
        
    Returns:
        The verified (or corrected) compressed resume.
    """
    verbose(f"  Stage 2: Verifying compressed resume accuracy...")
    
    prompt = RESUME_VERIFICATION_PROMPT.format(
        full_resume=full_resume,
        condensed_resume=compressed_resume
    )
    
    response = provider.generate(prompt)
    response = response.strip()
    
    # Check if the response indicates accuracy
    if response.upper().startswith("ACCURATE"):
        log_debug("  Verification: Condensed resume is accurate")
        return compressed_resume
    
    # The LLM indicated corrections are needed
    if response.upper().startswith("NEEDS_CORRECTION"):
        verbose("  Verification: Corrections needed, applying...")
        
        # Extract the corrected version (everything after "NEEDS_CORRECTION" line)
        lines = response.split("\n", 1)
        if len(lines) > 1:
            corrected = lines[1].strip()
            corrected = _strip_code_fences(corrected)
            
            if len(corrected) > 2400:
                info(f"  Warning: Corrected resume is {len(corrected)} chars (target: 2400)")
            else:
                log_debug(f"  Corrected compressed resume: {len(corrected)} chars")
            
            return corrected
        else:
            info("  Warning: LLM indicated corrections needed but provided none, using original")
            return compressed_resume
    
    # Unexpected response format - log and return original
    info("  Warning: Unexpected verification response format, using original compressed version")
    log_debug(f"  Response preview: {response[:200]}...")
    return compressed_resume


def get_resume_for_matching(
    resume_path: str | Path, 
    provider: "LLMProvider",
    *,
    force_refresh: bool = False,
) -> tuple[str, str]:
    """Get the compressed resume for job matching, using cache if valid.
    
    This function:
    1. Loads the full resume text
    2. Computes its checksum
    3. Checks if a valid cache exists (matching checksum)
    4. If cache valid, returns cached payload
    5. If cache invalid/missing, compresses via LLM and caches result
    
    Args:
        resume_path: Path to the resume file.
        provider: LLM provider for compression.
        force_refresh: If True, bypass cache and recompress.
        
    Returns:
        Tuple of (compressed_resume, full_resume_text).
        The compressed resume is used for job matching,
        the full text is available if needed.
    """
    # Load full resume
    full_resume = load_resume_text(resume_path)
    if not full_resume:
        return "", ""
    
    # Compute checksum
    checksum = compute_resume_checksum(full_resume)
    log_debug(f"Resume checksum: {checksum[:16]}...")
    
    # Check cache
    if not force_refresh:
        cache = load_resume_cache()
        if cache and cache.get("checksum") == checksum:
            verbose(f"  Using cached compressed resume (from {cache.get('timestamp', 'unknown')})")
            return cache["payload"], full_resume
        elif cache:
            verbose("  Resume changed - recompressing...")
        else:
            verbose("  No cached resume found - compressing...")
    else:
        verbose("  Force refresh requested - recompressing...")
    
    # Compress and cache
    compressed = compress_resume(full_resume, provider)
    save_resume_cache(checksum, compressed)
    
    return compressed, full_resume
def build_exclusion_rules(config: dict) -> str:
    """Build dynamic exclusion rules section from config lists.
    
    Generates clear, explicit instructions for the LLM based on user-configured
    exclusion lists. These are defense-in-depth: most are also enforced as
    pre-filters, but the LLM should catch anything that slips through.
    """
    rules = []
    
    # Blocked companies (aggregators/repost sites)
    blocked_companies = config.get("blocked_companies") or []
    if blocked_companies:
        company_list = ", ".join(blocked_companies)
        rules.append(f"""BLOCKED COMPANIES (known aggregator/repost sites — REJECT immediately):
  {company_list}
  
  These are resume-farming sites that scrape real job postings and repost them.
  If the employer name matches any of these, EXCLUDE the job.
  Also EXCLUDE any job where the employer appears to be a generic aggregator but
  the description clearly describes a role at a different well-known company
  (e.g., employer is "Flexionis" but description says "at Netflix").""")
    
    # Target countries
    target_countries = config.get("target_countries") or []
    if target_countries:
        country_list = ", ".join(target_countries)
        rules.append(f"""TARGET COUNTRIES (reject jobs outside these):
  {country_list}
  
  Only include jobs located in or remote-eligible for these countries.
  If a job is tagged with a country NOT in this list, EXCLUDE it.""")
    
    # Staffing agencies
    if config.get("filter_staffing_agencies", False):
        agency_names = config.get("staffing_agency_names") or []
        agency_str = ", ".join(agency_names) if agency_names else "(auto-detected)"
        rules.append(f"""STAFFING AGENCIES (reject unless role details are exceptionally clear):
  Known agencies: {agency_str}
  
  Reject postings from staffing firms, recruiting agencies, or talent marketplaces.
  Indicators: "on behalf of our client", "W2/C2C", "contract-to-hire", vague role
  descriptions with no named end client.""")
    
    # Excluded technologies
    exclude_techs = config.get("exclude_technologies") or []
    if exclude_techs:
        tech_list = ", ".join(exclude_techs)
        rules.append(f"""EXCLUDED TECHNOLOGIES (reject even if on my resume):
  {tech_list}
  
  If a job REQUIRES any of these technologies, EXCLUDE IT immediately.
  - Job title mentions these → EXCLUDE
  - Requirements list these → EXCLUDE  
  - Description says "strong [tech]" or "[tech] required" → EXCLUDE
  - "N+ years of [tech]" for these → EXCLUDE""")
    
    # Excluded role types
    exclude_roles = config.get("exclude_role_types") or []
    if exclude_roles:
        role_list = ", ".join(f'"{r}"' for r in exclude_roles)
        rules.append(f"""EXCLUDED ROLE TYPES (reject based on job title):
  {role_list}
  
  If the job title contains ANY of these terms (case-insensitive), EXCLUDE IT.
  Examples: "Full-Stack Engineer" → EXCLUDE, "Senior Frontend Developer" → EXCLUDE""")
    
    # Excluded keywords
    exclude_keywords = config.get("exclude_keywords") or []
    if exclude_keywords:
        kw_list = ", ".join(f'"{k}"' for k in exclude_keywords)
        rules.append(f"""EXCLUDED KEYWORDS (reject if found in description):
  {kw_list}
  
  If the job description contains ANY of these phrases, EXCLUDE IT.""")
    
    if not rules:
        return "No additional exclusion rules configured."
    
    return "\n\n".join(rules)


def build_scoring_guidance() -> str:
    """Build scoring calibration instructions for the LLM.
    
    These are generic quality-check instructions that apply regardless of the
    user's specific profile. They address common failure modes in job-matching:
    surface-level keyword matching, aggregator reposts, and domain mismatches.
    """
    return """
============================================================
SCORING CALIBRATION — READ BEFORE SCORING ANY JOB
============================================================

Your scoring must go BEYOND surface-level keyword overlap. Apply these checks:

1. AGGREGATOR / REPOST DETECTION:
   If the employer name looks like a generic staffing or aggregator company but the
   job description clearly describes a role at a DIFFERENT well-known company,
   this is a scraped repost. REJECT it (score 0). Signs of reposts:
   - Employer name doesn't match the company described in the body
   - Apply link domain doesn't match the employer
   - Description says "About [Famous Company]" but employer is something else

2. DOMAIN EXPERTISE FIT (most important calibration):
   Do NOT give high scores based on programming language overlap alone.
   The DOMAIN and CONTEXT of the work must match the candidate's background.
   Examples of BAD matches despite keyword overlap:
   - "Python" in QA automation ≠ "Python" in backend architecture
   - "C++" in embedded/RTOS ≠ "C++" in distributed systems
   - "Systems Engineer" in aerospace/defense ≠ "Systems Engineer" in software
   - "Data Engineer" doing ETL scripts ≠ "Data Platform Engineer" building infra
   Ask: "Would this candidate's actual work history prepare them for THIS role's
   day-to-day responsibilities?" If not, score LOW.

3. AI/ML ROLE CALIBRATION (prevents over-weighting):
   The candidate has MODERATE AI tooling experience (API integration, RAG pipelines,
   vector search) but is NOT an AI/ML specialist. AI-focused roles should NOT
   automatically score higher than non-AI backend/systems roles. Apply this test:
   - If the role's PRIMARY function is AI/ML (model training, fine-tuning, ML ops,
     ML research, prompt engineering as the main job) → score LOW, not a fit.
   - If the role uses AI as a TOOL within a backend/systems/infrastructure context
     → score normally based on the backend/systems fit.
   - NEVER boost a role's score simply because it mentions AI, LLM, or ML.
     A pure backend/systems role with strong tech-stack overlap should outscore an
     AI-titled role where the candidate's core backend skills are secondary.

4. UNREALISTIC REQUIREMENTS CHECK:
   If a job requires specific domain expertise, certifications, or clearances that
   are NOT evident in the candidate's resume, score it LOW regardless of tool overlap.
   Examples: security clearance, specific industry certifications, niche domain
   knowledge (aerospace, biotech, quant finance) that doesn't appear in the resume.

5. SENIORITY ALIGNMENT:
   A Staff/Principal role should not score high for a mid-level candidate, and vice
   versa. Check years-of-experience requirements against the resume timeline.

6. SCORE MEANING:
   95-100: Near-perfect fit — skills, domain, seniority, and tech stack all align
   90-94:  Strong fit — most requirements match with minor gaps
   85-89:  Good fit — solid overlap but some requirements are a stretch
   Below 85: Do not include in results
"""


def extract_whitelisted_job_fields(job: dict) -> dict:
    """Extract only the whitelisted fields from a job for LLM analysis.
    
    This reduces the amount of data sent to the LLM, lowering costs while
    preserving the information needed for job matching and quality checks.
    """
    highlights = job.get("job_highlights") or {}
    
    return {
        "job_id": job.get("job_id"),
        "job_title": job.get("job_title"),
        "employer_name": job.get("employer_name"),
        "job_publisher": job.get("job_publisher"),
        "job_employment_type": job.get("job_employment_type"),
        "job_apply_link": job.get("job_apply_link"),
        "job_description": job.get("job_description"),
        "job_is_remote": job.get("job_is_remote"),
        "qualifications": highlights.get("Qualifications"),
        "responsibilities": highlights.get("Responsibilities"),
    }


def build_jobs_listings_text(jobs: list) -> str:
    """Build the whitelisted job listings block for prompts."""
    jobs_text = ""
    for job in jobs:
        whitelisted = extract_whitelisted_job_fields(job)
        jobs_text += f"\n--- JOB: {whitelisted.get('job_id')} ---\n"
        jobs_text += f"Job ID: {whitelisted.get('job_id')}\n"
        jobs_text += f"Title: {whitelisted.get('job_title')}\n"
        jobs_text += f"Employer: {whitelisted.get('employer_name')}\n"
        if whitelisted.get("job_publisher"):
            jobs_text += f"Publisher: {whitelisted.get('job_publisher')}\n"
        jobs_text += f"Employment Type: {whitelisted.get('job_employment_type')}\n"
        jobs_text += f"Application URL: {whitelisted.get('job_apply_link') or ''}\n"
        jobs_text += f"Remote: {whitelisted.get('job_is_remote')}\n"
        if whitelisted.get("qualifications"):
            jobs_text += f"Qualifications: {whitelisted.get('qualifications')}\n"
        if whitelisted.get("responsibilities"):
            jobs_text += f"Responsibilities: {whitelisted.get('responsibilities')}\n"
        jobs_text += f"Description: {whitelisted.get('job_description')}\n"
    return jobs_text


def cot_batch_output_instruction() -> str:
    """Chain-of-thought JSON schema for per-job evaluation (batch or single-pass)."""
    return """
============================================================
IMPORTANT: OUTPUT FORMAT (CHAIN OF THOUGHT — REQUIRED)
============================================================

Return ONLY a JSON **object** (not a bare array) with exactly two keys:

1. "batch_summary": A concise paragraph (4-8 sentences) summarizing your analysis of this
   batch. MUST include:
   - How many of the jobs were plausible matches vs. clearly unqualified
   - The most common reasons jobs were eliminated (e.g., "14 required Go/Rust",
     "9 were frontend roles", "6 were from staffing agencies")
   - Whether there were near-miss jobs that almost qualified (scored 80-84) and roughly
     how many. Briefly note what kept them out (e.g., "3 near-misses: 2 needed Kubernetes
     experience, 1 was too junior")
   - Your overall assessment: was this a strong, average, or weak pool for this candidate?
     Were there clearly MORE good matches than the result limit, or was the pool thin?

2. "jobs": A JSON array with EXACTLY one object for EACH job in the listings below
   (same count as jobs). Evaluate every job; do not skip jobs.

Each object in the "jobs" array MUST have these fields:
- "job_id": The exact job_id string from the job listing (preserve the == suffix if present)
- "pass_1_reasoning": Brief text. FIRST apply exclusion/negative constraints and legitimacy
  checks in plain language (e.g. "Posting requires Go; Go is excluded → eliminate.") BEFORE scoring.
- "survives_pass_1": true if the job passes exclusions and is worth scoring; false if it must be rejected.
- "score": If survives_pass_1 is false, use 0. If true, integer 85-100 per SCORING CALIBRATION.
- "reason": If eliminated, short elimination reason. If survived, 1-2 sentences on why this score (fit only).
- "job_short_desc": If survives_pass_1 is true, exactly 8 complete sentences describing day-to-day duties
  from the posting (not scoring). If false, use "".

Example response:
{"batch_summary":"Of the 30 jobs reviewed, 8 were plausible backend/systems matches. 12 were eliminated for requiring excluded technologies (7 Go, 3 Java, 2 Ruby). 6 were frontend/full-stack roles. 4 were staffing agency reposts. 2 near-misses scored 82-83 (one needed Kubernetes, one was junior-level). Overall a below-average pool — the candidate's Python/C++ backend profile had limited overlap with this batch's heavy Go/Java skew.","jobs":[{"job_id":"x==","pass_1_reasoning":"Requires Go; Go is excluded.","survives_pass_1":false,"score":0,"reason":"Eliminated: requires Go","job_short_desc":""},{"job_id":"y==","pass_1_reasoning":"No excluded stack; backend role; not an aggregator repost.","survives_pass_1":true,"score":93,"reason":"Strong Python/backend fit.","job_short_desc":"Sentence one. ... Sentence eight."}]}

Return ONLY this JSON object — no markdown fences, no prose outside the object.
"""


def final_round_output_instruction(max_results: int) -> str:
    """Output format for the tournament final (top candidates from each batch)."""
    return f"""
============================================================
IMPORTANT: FINAL ROUND — OUTPUT FORMAT
============================================================

You are given the TOP candidates from each prior batch (already scored). Using the resume and
SCORING CALIBRATION, select the OVERALL best matches.

Return ONLY a JSON **object** (not a bare array) with exactly two keys:

1. "batch_summary": A concise paragraph (3-6 sentences) providing a FINAL OVERALL assessment:
   - How many of the finalists were strong vs. marginal matches
   - Whether the final set represents a strong, average, or weak set of opportunities
   - Any patterns across the finalists (e.g., "most are cloud infrastructure roles",
     "heavy skew toward observability platforms")
   - If fewer than {max_results} jobs made the cut, explain why (thin pool, near-misses, etc.)

2. "jobs": A JSON array with AT MOST {max_results} objects (the single best overall matches).
   Each object MUST have:
   - "job_id": exact string from the candidate list
   - "score": integer 85-100
   - "reason": 1-2 sentences final fit rationale
   - "job_short_desc": exactly 8 complete sentences on duties (refine or keep consistent with prior text)

   Do not include pass_1_reasoning or survives_pass_1 in this round.

Return ONLY this JSON object — no markdown, no extra text.
"""


def batch_jobs(jobs: list, target_batch_size: int = DEFAULT_JOBS_PER_LLM_BATCH) -> list[list]:
    """Split jobs into batches of approximately equal size, targeting target_batch_size per batch."""
    n = len(jobs)
    if n <= target_batch_size:
        return [jobs]
    num_batches = max(1, math.ceil(n / target_batch_size))
    chunk = math.ceil(n / num_batches)
    return [jobs[i : i + chunk] for i in range(0, n, chunk)]


def build_prompt(jobs: list, config: dict, resume_text: str) -> str:
    """Build the analysis prompt for a single LLM call (typically <= provider jobs_per_llm_batch).

    Order: instructions, scoring guidance, resume (before jobs), output format, job listings, reminder.
    """
    jobs_text = build_jobs_listings_text(jobs)
    exclusion_rules = build_exclusion_rules(config)
    base_prompt = (
        (config.get("prompt") or "")
        .replace("{job_count}", str(len(jobs)))
        .replace("{exclusion_rules}", exclusion_rules)
    )
    resume_block = resume_text if resume_text else "[No resume text provided]"
    scoring_guidance = build_scoring_guidance()
    output_instruction = cot_batch_output_instruction()

    prompt = f"""
{base_prompt}

{scoring_guidance}

=== REFERENCE: MY RESUME (use ONLY to assess fit — NOT a job listing) ===
{resume_block}
=== END OF RESUME ===

{output_instruction}

=== JOB LISTINGS TO EVALUATE ===
{jobs_text}
=== END OF JOB LISTINGS ===

CRITICAL REMINDER:
- Return ONLY the JSON object described above (with "batch_summary" and "jobs" keys)
- The "jobs" array must have chain-of-thought fields for every job
- Use the EXACT job_id from the listings (including any == suffix)
- Apply negative constraints in pass_1_reasoning before assigning survives_pass_1
- Apply SCORING CALIBRATION — reject surface-level keyword matches
""".strip()

    return prompt


def build_batch_prompt(
    jobs: list,
    config: dict,
    resume_text: str,
    batch_num: int,
    total_batches: int,
) -> str:
    """Prompt for one batch of the multi-batch pipeline (evaluate all jobs, CoT, then caller keeps top N)."""
    jobs_text = build_jobs_listings_text(jobs)
    exclusion_rules = build_exclusion_rules(config)
    base_prompt = (
        (config.get("prompt") or "")
        .replace("{job_count}", str(len(jobs)))
        .replace("{exclusion_rules}", exclusion_rules)
    )
    resume_block = resume_text if resume_text else "[No resume text provided]"
    scoring_guidance = build_scoring_guidance()
    output_instruction = cot_batch_output_instruction()

    header = f"""
=== BATCH CONTEXT ===
This is batch {batch_num} of {total_batches}. Jobs in this batch: {len(jobs)}.
Evaluate ONLY the jobs listed below in this batch. Other batches are evaluated separately.
"""

    prompt = f"""
{base_prompt}

{header}

{scoring_guidance}

=== REFERENCE: MY RESUME (use ONLY to assess fit — NOT a job listing) ===
{resume_block}
=== END OF RESUME ===

{output_instruction}

=== JOB LISTINGS TO EVALUATE (THIS BATCH ONLY) ===
{jobs_text}
=== END OF JOB LISTINGS ===

CRITICAL REMINDER:
- Return ONLY the JSON object (with "batch_summary" and "jobs" keys)
- The "jobs" array must have one object per job in THIS batch (chain-of-thought fields required)
- Use the EXACT job_id from the listings (including any == suffix)
- Apply negative constraints in pass_1_reasoning before assigning survives_pass_1
""".strip()

    return prompt


def build_final_prompt(
    batch_winners: list[dict],
    config: dict,
    resume_text: str,
    jobs_by_id: dict[str, dict],
    *,
    max_results: int = MAX_LLM_JOBS_PER_ROUND,
) -> str:
    """Tournament final: choose overall top matches from per-batch winners."""
    exclusion_rules = build_exclusion_rules(config)
    base_prompt = (
        (config.get("prompt") or "")
        .replace("{job_count}", str(len(batch_winners)))
        .replace("{exclusion_rules}", exclusion_rules)
    )
    resume_block = resume_text if resume_text else "[No resume text provided]"
    scoring_guidance = build_scoring_guidance()
    out_fmt = final_round_output_instruction(max_results)

    lines: list[str] = []
    for i, row in enumerate(batch_winners, start=1):
        jid = row.get("job_id")
        full = jobs_by_id.get(jid) or {}
        title = full.get("job_title") or ""
        employer = full.get("employer_name") or ""
        lines.append(f"\n--- CANDIDATE {i} ---")
        lines.append(f"job_id: {jid}")
        lines.append(f"Title: {title}")
        lines.append(f"Employer: {employer}")
        lines.append(f"Prior score (batch round): {row.get('score')}")
        lines.append(f"Prior reason: {row.get('reason', '')}")
        lines.append(f"Prior job_short_desc: {row.get('job_short_desc', '')}")
    candidates_text = "\n".join(lines)

    prompt = f"""
{base_prompt}

{scoring_guidance}

=== REFERENCE: MY RESUME (use ONLY to assess fit — NOT a job listing) ===
{resume_block}
=== END OF RESUME ===

{out_fmt}

=== FINALIST CANDIDATES (TOP FROM EACH BATCH — PICK OVERALL BEST) ===
{candidates_text}
=== END OF FINALISTS ===

CRITICAL REMINDER:
- Return ONLY the JSON object (with "batch_summary" and "jobs" keys)
- The "jobs" array must have at most {max_results} entries
- Use EXACT job_id values from the candidate list
- You may adjust scores and reasons versus the batch round; justify in "reason"
""".strip()

    return prompt


def parse_llm_job_scores(llm_response: str) -> tuple[list[dict], str]:
    """Parse the LLM response to extract job score objects and an optional batch summary.

    Handles two formats:
    - New: ``{"batch_summary": "...", "jobs": [...]}``
    - Legacy: bare JSON array ``[...]``

    Returns:
        Tuple of (job_rows, batch_summary). batch_summary is "" when absent.
    """
    text = llm_response.strip()

    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if "```" in text:
            text = text.rsplit("```", 1)[0]
        text = text.strip()

    # Try parsing as a JSON object with "jobs" key (new format)
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "jobs" in obj:
                jobs = obj["jobs"] if isinstance(obj["jobs"], list) else []
                summary = obj.get("batch_summary", "")
                return jobs, str(summary)
        except json.JSONDecodeError:
            # Try to extract the outermost { ... }
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    obj = json.loads(text[start : end + 1])
                    if isinstance(obj, dict) and "jobs" in obj:
                        jobs = obj["jobs"] if isinstance(obj["jobs"], list) else []
                        summary = obj.get("batch_summary", "")
                        return jobs, str(summary)
                except json.JSONDecodeError:
                    pass

    # Legacy fallback: bare JSON array
    if not text.startswith("["):
        # Maybe the object parse failed but there's still a { ... } with jobs inside
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                obj = json.loads(text[brace_start : brace_end + 1])
                if isinstance(obj, dict) and "jobs" in obj:
                    jobs = obj["jobs"] if isinstance(obj["jobs"], list) else []
                    summary = obj.get("batch_summary", "")
                    return jobs, str(summary)
            except json.JSONDecodeError:
                pass

        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            text = text[start : end + 1]

    try:
        results = json.loads(text)
        if isinstance(results, list):
            return results, ""
    except json.JSONDecodeError:
        pass

    return [], ""


def normalize_scored_jobs(
    results: list[dict],
    *,
    apply_cot_filter: bool = True,
    max_results: int | None = None,
) -> list[dict]:
    """Filter and rank LLM job rows for merge_scores_with_jobs.

    - If apply_cot_filter: drop survives_pass_1 false; require score >= 85 for survivors.
    - If a row has no survives_pass_1 (legacy), keep only if score >= 85.
    - Sorted by score descending; optionally capped to max_results.
    """
    kept: list[dict] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        job_id = row.get("job_id")
        score = row.get("score")
        if job_id is None or score is None:
            continue
        try:
            score_int = int(score)
        except (TypeError, ValueError):
            continue

        if apply_cot_filter and "survives_pass_1" in row:
            if row.get("survives_pass_1") is False:
                continue
        if score_int < 85:
            continue

        out = dict(row)
        out["score"] = score_int
        kept.append(out)

    kept.sort(key=lambda x: int(x.get("score", 0)), reverse=True)
    if max_results is not None:
        kept = kept[:max_results]
    return kept


def analyze_jobs(
    jobs: list,
    provider: LLMProvider,
    config: dict,
    resume_text: str,
    *,
    save_debug_files: bool = False,
    debug_tag: str | None = None,
) -> str:
    """Send one batch of jobs to the LLM (single-shot analysis)."""
    verbose(f"Sending {len(jobs)} jobs to {provider.name} ({provider.model})")

    prompt = build_prompt(jobs, config, resume_text)
    log_debug(f"  Prompt length: {len(prompt)} chars")

    if save_debug_files:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = f"-{debug_tag}" if debug_tag else ""
        debug_filename = f"llmquery-debug-{timestamp}{suffix}.json"
        debug_data = {
            "timestamp": timestamp,
            "provider": provider.name,
            "model": provider.model,
            "job_count": len(jobs),
            "prompt_length_chars": len(prompt),
            "prompt": prompt,
        }
        Path(debug_filename).write_text(json.dumps(debug_data, indent=2), encoding="utf-8")
        log_debug(f"  Saved LLM query to {debug_filename}")

    return provider.generate(prompt)


def analyze_jobs_batched(
    jobs: list,
    provider: LLMProvider,
    config: dict,
    resume_text: str,
    *,
    save_debug_files: bool = False,
) -> tuple[list[dict], list[str]]:
    """Batch pipeline: equal-ish batches (~provider.jobs_per_llm_batch), top MAX per batch, then final.

    If only one batch is produced, the final tournament is skipped — the batch winners are final.

    Returns:
        Tuple of (final_job_results, batch_summaries). batch_summaries includes
        per-batch summaries and a final-round summary (if a final round ran).
    """
    batches = batch_jobs(jobs, target_batch_size=provider.jobs_per_llm_batch)
    total = len(batches)
    jobs_by_id = {j.get("job_id"): j for j in jobs if j.get("job_id")}

    batch_winners: list[dict] = []
    all_summaries: list[str] = []
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    verbose(f"Batched analysis: {len(jobs)} jobs in {total} batch(es)")
    for bi, batch in enumerate(batches, start=1):
        verbose(
            f"Batch {bi}/{total}: sending {len(batch)} jobs to "
            f"{provider.name} ({provider.model})"
        )
        prompt = build_batch_prompt(batch, config, resume_text, bi, total)
        log_debug(f"  Batch {bi}/{total} prompt length: {len(prompt)} chars")
        if save_debug_files:
            path = Path(f"llmquery-debug-{timestamp}-batch{bi}.json")
            path.write_text(
                json.dumps(
                    {
                        "timestamp": timestamp,
                        "batch": bi,
                        "total_batches": total,
                        "provider": provider.name,
                        "model": provider.model,
                        "job_count": len(batch),
                        "prompt_length_chars": len(prompt),
                        "prompt": prompt,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            log_debug(f"  Saved {path}")

        raw = provider.generate(prompt)
        parsed, summary = parse_llm_job_scores(raw)
        if summary:
            label = f"[Batch {bi}/{total}]" if total > 1 else "[Batch 1/1]"
            tagged = f"{label} {summary}"
            all_summaries.append(tagged)
            verbose(f"  Batch summary: {summary}")
        winners = normalize_scored_jobs(
            parsed, apply_cot_filter=True, max_results=MAX_LLM_JOBS_PER_ROUND
        )
        verbose(
            f"Batch {bi}/{total}: kept {len(winners)} jobs "
            f"(top {MAX_LLM_JOBS_PER_ROUND} after filter)"
        )
        batch_winners.extend(winners)

    if not batch_winners:
        return [], all_summaries

    if total == 1:
        verbose(
            f"Single batch only: skipping final tournament ({len(batch_winners)} jobs kept)."
        )
        return batch_winners, all_summaries

    verbose(
        f"Final round: {len(batch_winners)} finalists → "
        f"selecting top {MAX_LLM_JOBS_PER_ROUND}"
    )
    final_prompt = build_final_prompt(
        batch_winners, config, resume_text, jobs_by_id, max_results=MAX_LLM_JOBS_PER_ROUND
    )
    log_debug(f"  Final round prompt length: {len(final_prompt)} chars")
    if save_debug_files:
        fpath = Path(f"llmquery-debug-{timestamp}-final.json")
        fpath.write_text(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "stage": "final",
                    "provider": provider.name,
                    "model": provider.model,
                    "finalist_count": len(batch_winners),
                    "prompt_length_chars": len(final_prompt),
                    "prompt": final_prompt,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log_debug(f"  Saved {fpath}")

    final_raw = provider.generate(final_prompt)
    final_parsed, final_summary = parse_llm_job_scores(final_raw)
    if final_summary:
        all_summaries.append(f"[Final round] {final_summary}")
        verbose(f"  Final round summary: {final_summary}")
    # Final round has no survives_pass_1; do not drop rows missing that key
    final_results = normalize_scored_jobs(
        final_parsed, apply_cot_filter=False, max_results=MAX_LLM_JOBS_PER_ROUND
    )
    return final_results, all_summaries


def merge_scores_with_jobs(
    llm_results: list[dict],
    jobs: list[dict],
    *,
    markdown: bool = False,
    batch_summaries: list[str] | None = None,
) -> str:
    """Merge LLM scoring results back with full job data.
    
    The LLM returns only job_id, score, reason, and job_short_desc. All other fields
    (links, apply options, location, etc.) are sourced from the original
    JSearch job data by joining on job_id.
    
    Args:
        llm_results: List of dicts with job_id, score, reason, and job_short_desc from LLM.
        jobs: Full list of job dictionaries from JSearch.
        markdown: Whether to output markdown (True) or JSON (False).
        batch_summaries: Optional list of LLM-generated batch summaries.
        
    Returns:
        Complete analysis output with full job details and scores.
    """
    jobs_by_id = {job.get("job_id"): job for job in jobs if job.get("job_id")}
    
    matched_jobs = []
    
    for result in llm_results:
        job_id = result.get("job_id")
        score = result.get("score")
        reason = result.get("reason", "")
        
        if not job_id or score is None:
            continue
        
        full_job = jobs_by_id.get(job_id)
        if not full_job:
            log_debug(f"  Warning: job_id '{job_id}' from LLM not found in job list")
            continue
        job_short_desc = result.get("job_short_desc") or full_job.get("job_description") or ""
        
        matched_job = {
            "position": full_job.get("job_title"),
            "company": full_job.get("employer_name"),
            "score": score,
            "reason": reason,
            "requirements": _extract_requirements(full_job),
            "short_description": job_short_desc,
            "link": full_job.get("job_apply_link"),
            "job_apply_link": full_job.get("job_apply_link"),
            "apply_options": full_job.get("apply_options") or [],
            "apply_links": _collect_apply_links(full_job),
            "job_id": job_id,
            "employment_type": full_job.get("job_employment_type"),
            "is_remote": full_job.get("job_is_remote"),
            "location": full_job.get("job_location"),
            "city": full_job.get("job_city"),
            "state": full_job.get("job_state"),
            "country": full_job.get("job_country"),
        }
        matched_jobs.append(matched_job)
    
    # Sort by score descending
    matched_jobs.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    summaries = batch_summaries or []

    if markdown:
        return _format_jobs_as_markdown(matched_jobs, batch_summaries=summaries)
    else:
        output: dict[str, Any] = {}
        if summaries:
            output["batch_summaries"] = summaries
        output["jobs"] = matched_jobs
        return json.dumps(output, indent=2)


def _extract_requirements(job: dict) -> list[str]:
    """Extract key requirements from job highlights."""
    highlights = job.get("job_highlights") or {}
    qualifications = highlights.get("Qualifications") or []
    
    # Return first few qualifications as requirements
    if isinstance(qualifications, list):
        return qualifications[:5]  # Limit to first 5
    return []


def _collect_apply_links(job: dict) -> list[str]:
    """Collect all unique application links from primary and apply options."""
    links: list[str] = []
    seen: set[str] = set()

    primary = job.get("job_apply_link")
    if isinstance(primary, str) and primary.strip():
        cleaned = primary.strip()
        links.append(cleaned)
        seen.add(cleaned)

    apply_options = job.get("apply_options") or []
    if isinstance(apply_options, list):
        for option in apply_options:
            if not isinstance(option, dict):
                continue
            apply_link = option.get("apply_link")
            if not isinstance(apply_link, str):
                continue
            cleaned = apply_link.strip()
            if not cleaned or cleaned in seen:
                continue
            links.append(cleaned)
            seen.add(cleaned)

    return links


def _format_jobs_as_markdown(
    jobs: list[dict],
    *,
    batch_summaries: list[str] | None = None,
) -> str:
    """Format matched jobs as markdown output."""
    lines = []

    if batch_summaries:
        lines.append("# Pool Analysis")
        lines.append("")
        for summary in batch_summaries:
            lines.append(f"- {summary}")
        lines.append("")
        lines.append("---")
        lines.append("")

    for job in jobs:
        company = job.get("company", "Unknown Company")
        position = job.get("position", "Unknown Position")
        score = job.get("score", 0)
        reason = job.get("reason", "")
        link = job.get("link", "")
        apply_links = job.get("apply_links") or []
        requirements = job.get("requirements", [])
        short_desc = job.get("short_description", "")
        is_remote = job.get("is_remote", False)
        location = job.get("location") or ""
        
        lines.append(f"## {company} - {position}")
        lines.append("")
        lines.append(f"**Match Score:** {score}/100")
        lines.append("")
        lines.append(f"**Why it fits:** {reason}")
        lines.append("")
        
        if requirements:
            lines.append("**Requirements:**")
            for req in requirements:
                lines.append(f"- {req}")
            lines.append("")
        
        lines.append(f"**Remote:** {'Yes' if is_remote else 'No'}")
        if location:
            lines.append(f"**Location:** {location}")
        lines.append("")
        
        if short_desc:
            lines.append(f"**Description:** {short_desc}")
            lines.append("")
        
        if apply_links:
            lines.append("**Apply Links:**")
            for apply_link in apply_links:
                lines.append(f"- {apply_link}")
            lines.append("")
        elif link:
            lines.append(f"**[Apply Here]({link})**")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    return "\n".join(lines)


def clean_duplicate_output(analysis: str, *, markdown: bool = False) -> str:
    """Remove duplicate format sections when LLM outputs both markdown and JSON.
    
    Sometimes Gemini outputs both formats even when only one is requested.
    This extracts just the requested format.
    
    Args:
        analysis: The raw analysis text from the LLM.
        markdown: Whether markdown (True) or JSON (False) was requested.
    
    Returns:
        Cleaned output containing only the requested format.
    """
    text = analysis.strip()
    
    if markdown:
        # If JSON array appears in markdown output, strip it
        has_code_fence = '```' in text
        has_markdown_content = any(
            text.startswith(prefix) for prefix in ('*', '-', '#', '##', '###')
        )
        
        if has_code_fence and has_markdown_content:
            # Keep only the markdown portion (before code fences)
            parts = text.split('```')
            markdown_part = parts[0].strip()
            
            # If the markdown part has actual content, use it
            if markdown_part and len(markdown_part) > 50:
                return markdown_part
            
            # Otherwise check if markdown comes after the code block
            if len(parts) > 2:
                after_code = parts[2].strip() if len(parts) > 2 else ''
                if after_code and any(
                    after_code.startswith(p) for p in ('*', '-', '#', '##', '###')
                ):
                    return after_code
    else:
        # JSON mode: extract just the JSON array
        
        # If it already starts with a valid JSON array, return as-is
        if text.startswith('['):
            return text
        
        # Try to find and extract JSON array from mixed output
        start = text.find('[')
        end = text.rfind(']')
        
        if start >= 0 and end > start:
            potential_json = text[start:end + 1]
            
            # Validate it's actually parseable JSON before returning
            try:
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                pass  # Fall through to return original
        
        # Check for JSON in code fences
        if '```' in text:
            # Extract content between fences
            fence_start = text.find('```')
            fence_content_start = text.find('\n', fence_start) + 1
            fence_end = text.find('```', fence_content_start)
            
            if fence_content_start > 0 and fence_end > fence_content_start:
                fenced_content = text[fence_content_start:fence_end].strip()
                if fenced_content.startswith('['):
                    try:
                        json.loads(fenced_content)
                        return fenced_content
                    except json.JSONDecodeError:
                        pass
    
    return analysis


def validate_results(analysis: str, *, markdown: bool = False) -> str:
    """Filter out invalid job matches like resume content mistaken for jobs.
    
    Args:
        analysis: The raw analysis text from the LLM.
        markdown: Whether the output is markdown (True) or JSON (False).
    
    Returns:
        Cleaned analysis with invalid entries removed.
    """
    # Markers that indicate resume content or invalid entries
    invalid_markers = [
        'based on my work experience',
        'based on my resume',
        'this position is based on',
        'from my background',
        'my professional experience',
        'link:  (',  # Empty link with parenthetical
        'link: (',
        'link: n/a',
        'link: none',
        'link: not available',
        'no link provided',
        'no application link',
    ]
    
    if markdown:
        # For markdown, remove sections with invalid content
        lines = analysis.split('\n')
        filtered_lines = []
        skip_section = False
        current_section_lines: list[str] = []
        
        for line in lines:
            line_lower = line.lower()
            
            # Check if this is a section header (new job entry)
            is_section_start = (
                line.startswith('### ') or 
                line.startswith('## ') or
                line.startswith('---')
            )
            
            if is_section_start:
                # Flush previous section if valid
                if current_section_lines and not skip_section:
                    filtered_lines.extend(current_section_lines)
                
                # Start new section
                current_section_lines = [line]
                skip_section = False
            else:
                current_section_lines.append(line)
                
                # Check for invalid markers in this line
                if any(marker in line_lower for marker in invalid_markers):
                    skip_section = True
        
        # Flush final section
        if current_section_lines and not skip_section:
            filtered_lines.extend(current_section_lines)
        
        return '\n'.join(filtered_lines)
    
    else:
        # For JSON, parse and filter
        try:
            text = analysis.strip()
            
            # Strip markdown code fences if present (handle ```json, ```, etc.)
            if text.startswith('```'):
                # Remove opening fence (with optional language tag)
                first_newline = text.find('\n')
                if first_newline != -1:
                    text = text[first_newline + 1:]
                # Remove closing fence
                if '```' in text:
                    text = text.rsplit('```', 1)[0]
                text = text.strip()
            
            jobs = json.loads(text)
            
            if not isinstance(jobs, list):
                return analysis  # Unexpected format, return as-is
            
            valid_jobs = []
            for job in jobs:
                link = job.get('link', '') or ''
                # Check for title/position (prompt asks for "position" but also accept "title" for compatibility)
                title = job.get('position', '') or job.get('title', '') or job.get('job_title', '') or ''
                
                # Must have a valid HTTP(S) link
                if not (link.startswith('http://') or link.startswith('https://')):
                    continue
                
                # Check for invalid markers in any field
                job_text = json.dumps(job).lower()
                if any(marker in job_text for marker in invalid_markers):
                    continue
                
                # Skip entries with empty or placeholder titles
                if not title or title.lower() in ('n/a', 'none', 'unknown', ''):
                    continue
                
                valid_jobs.append(job)
            
            return json.dumps(valid_jobs, indent=2)
        
        except (json.JSONDecodeError, TypeError, KeyError):
            return analysis  # Return as-is if parsing fails
