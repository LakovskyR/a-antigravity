"""
AI Commentator â€” Module 3
Routes statistical results to selected LLM for dual-persona commentary.
Outputs: final_delivery_payload.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import requests
from dotenv import load_dotenv


def load_env():
    """Load environment variables."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def load_config():
    """Load run configuration."""
    config_path = Path(__file__).parent.parent / "tmp" / "run_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config):
    """Save updated configuration."""
    config_path = Path(__file__).parent.parent / "tmp" / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_statistical_results():
    """Load statistical results from previous module."""
    results_path = Path(__file__).parent.parent / "tmp" / "statistical_results.json"
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_final_payload(payload: dict):
    """Save final delivery payload."""
    tmp_dir = Path(__file__).parent.parent / "tmp"
    payload_path = tmp_dir / "final_delivery_payload.json"
    
    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    
    print(f"   ðŸ’¾ Saved: {payload_path}")
    return payload_path


def create_commentary_prompt(statistical_results: dict) -> str:
    """
    Create a prompt that requests both academic and business commentary.
    Only aggregated statistical data is sent â€” no raw survey responses.
    """
    # Extract summary statistics only (not raw data)
    summary = {
        "project": statistical_results.get("project_metadata", {}),
        "modules_run": list(statistical_results.get("results", {}).keys()),
        "key_findings": extract_key_findings(statistical_results)
    }
    
    prompt = f"""You are an expert survey analytics consultant. Analyze the following statistical results and provide commentary in TWO distinct personas.

STATISTICAL SUMMARY:
{json.dumps(summary, indent=2)}

YOUR TASK:
Provide analysis in two distinct sections:

## ACADEMIC/STATISTICAL PERSPECTIVE
Focus on:
- Methodological rigor and validity
- Statistical significance and confidence intervals
- Sample size considerations
- Potential biases or limitations
- Comparison to established research standards

## BUSINESS/MARKETING PERSPECTIVE  
Focus on:
- Strategic implications for the business
- Actionable recommendations
- Client-facing language and narrative
- Competitive positioning insights
- Next steps and opportunities

RULES:
- Be concise but thorough (3-5 bullet points per section)
- Do NOT hallucinate data â€” only reference what's in the summary
- Flag any concerns about data quality or interpretation
- Highlight the most important insight first in each section

Respond in JSON format:
{{
  "academic_commentary": "string",
  "business_commentary": "string",
  "confidence_level": "high|medium|low",
  "key_recommendations": ["string"]
}}
"""
    return prompt


def extract_key_findings(statistical_results: dict) -> dict:
    """Extract key findings for the LLM prompt (aggregated data only)."""
    findings = {}
    results = statistical_results.get("results", {})
    
    # Descriptive stats summary
    if "descriptive" in results:
        desc = results["descriptive"]
        findings["variables_analyzed"] = len(desc.get("frequencies", {}))
    
    # Typology summary
    if "typology" in results:
        typo = results["typology"]
        if "error" not in typo:
            findings["clusters_found"] = typo.get("optimal_k")
            findings["cluster_quality"] = typo.get("silhouette_score")
    
    # Modeling summary
    if "modeling" in results:
        model = results["modeling"]
        if "error" not in model:
            findings["best_model"] = model.get("winner")
            findings["model_performance"] = model.get("winner_score")
            if "shap_summary" in model:
                findings["top_predictors"] = model["shap_summary"].get("top_features", [])
    
    # Regression summary
    if "regression" in results:
        reg = results["regression"]
        if "error" not in reg:
            findings["significant_predictors"] = [
                c["variable"] for c in reg.get("coefficients", [])
                if c.get("ci_95_low") and c.get("ci_95_high") and 
                   (c["ci_95_low"] > 0 or c["ci_95_high"] < 0)
            ][:5]
    
    return findings


def call_claude(prompt: str, api_key: str) -> Optional[dict]:
    """Call Claude API for commentary."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    data = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        content = response.json()["content"][0]["text"]
        
        # Try to parse JSON response
        try:
            # Extract JSON if wrapped in code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content.strip())
        except:
            # Return as raw text if not valid JSON
            return {
                "academic_commentary": content,
                "business_commentary": content,
                "raw_response": content
            }
    except Exception as e:
        print(f"      âŒ Claude API error: {e}")
        return None


def call_openai(prompt: str, api_key: str) -> Optional[dict]:
    """Call OpenAI API for commentary."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an expert survey analytics consultant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1500
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        content = response.json()["choices"][0]["message"]["content"]
        
        # Try to parse JSON
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content.strip())
        except:
            return {
                "academic_commentary": content,
                "business_commentary": content,
                "raw_response": content
            }
    except Exception as e:
        print(f"      âŒ OpenAI API error: {e}")
        return None


def call_kimi(prompt: str, api_key: str) -> Optional[dict]:
    """Call Kimi API for commentary."""
    url = "https://api.moonshot.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "moonshot-v1-8k",
        "messages": [
            {"role": "system", "content": "You are an expert survey analytics consultant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1500
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        content = response.json()["choices"][0]["message"]["content"]
        
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content.strip())
        except:
            return {
                "academic_commentary": content,
                "business_commentary": content,
                "raw_response": content
            }
    except Exception as e:
        print(f"      âŒ Kimi API error: {e}")
        return None


def call_perplexity(prompt: str, api_key: str) -> Optional[dict]:
    """Call Perplexity API for commentary."""
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "You are an expert survey analytics consultant with access to external research."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1500
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        content = response.json()["choices"][0]["message"]["content"]
        
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content.strip())
        except:
            return {
                "academic_commentary": content,
                "business_commentary": content,
                "raw_response": content
            }
    except Exception as e:
        print(f"      âŒ Perplexity API error: {e}")
        return None


def generate_commentary(statistical_results: dict, provider: str) -> dict:
    """Generate AI commentary using selected provider."""
    print(f"\n   ðŸ¤– Requesting commentary from {provider.upper()}...")
    
    prompt = create_commentary_prompt(statistical_results)
    
    # Route to selected provider
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    
    if not api_key:
        print(f"      âš ï¸  {provider} API key not configured")
        return generate_fallback_commentary(statistical_results)
    
    provider_functions = {
        "claude": call_claude,
        "openai": call_openai,
        "kimi": call_kimi,
        "perplexity": call_perplexity
    }
    
    if provider in provider_functions:
        result = provider_functions[provider](prompt, api_key)
        if result:
            print(f"      âœ… Commentary received")
            return result
    
    # Fallback
    print(f"      âš ï¸  Falling back to default commentary")
    return generate_fallback_commentary(statistical_results)


def generate_fallback_commentary(statistical_results: dict) -> dict:
    """Generate placeholder commentary when LLM is unavailable."""
    results = statistical_results.get("results", {})
    modules = list(results.keys())
    
    return {
        "academic_commentary": f"Analysis completed using {', '.join(modules)}. Statistical methods applied per standard survey research protocols. Review confidence intervals and sample sizes for interpretation.",
        "business_commentary": f"Key insights derived from {', '.join(modules)} analysis. Recommend follow-up research to validate findings and develop actionable strategies.",
        "confidence_level": "medium",
        "key_recommendations": [
            "Review detailed statistical outputs",
            "Validate findings with additional data collection",
            "Develop implementation roadmap based on insights"
        ],
        "note": "Fallback commentary (LLM unavailable)"
    }


def format_for_powerbi(payload: dict) -> str:
    """
    Format payload as plain text for Power BI Power Query integration.
    Returns a synthesis per analysis module.
    """
    lines = []
    meta = payload.get("project_metadata", {})
    
    lines.append(f"PROJECT: {meta.get('project_name', 'Unknown')}")
    lines.append(f"WAVE: {meta.get('wave', 'Unknown')}")
    lines.append("")
    
    ai_insights = payload.get("ai_insights", {})
    lines.append("ACADEMIC PERSPECTIVE:")
    lines.append(ai_insights.get("academic_commentary", "N/A"))
    lines.append("")
    
    lines.append("BUSINESS PERSPECTIVE:")
    lines.append(ai_insights.get("business_commentary", "N/A"))
    
    return "\n".join(lines)


def main():
    """Main AI commentary workflow."""
    parser = argparse.ArgumentParser(description="AI Commentator for Antigravity")
    parser.add_argument("--mode", choices=["default", "powerbi_table"], 
                       default="default", help="Output mode")
    args = parser.parse_args()
    
    # Power BI mode
    if args.mode == "powerbi_table":
        payload_path = Path(__file__).parent.parent / "tmp" / "final_delivery_payload.json"
        if payload_path.exists():
            with open(payload_path, "r") as f:
                payload = json.load(f)
            print(format_for_powerbi(payload))
        else:
            print("No analysis results available")
        return
    
    # Standard mode
    print("=" * 70)
    print("ðŸ’¬ MODULE 3: AI COMMENTATOR")
    print("=" * 70)
    
    load_env()
    config = load_config()
    statistical_results = load_statistical_results()
    
    # Update status
    config["run_info"]["status"] = "commentary"
    save_config(config)
    
    provider = config["project_metadata"]["llm_provider"]
    print(f"\nðŸŽ¯ Provider: {provider}")
    
    # Generate commentary
    commentary = generate_commentary(statistical_results, provider)
    
    # Build final payload
    final_payload = {
        "project_metadata": statistical_results["project_metadata"],
        "run_info": {
            **statistical_results["run_info"],
            "commentary_generated_at": datetime.now().isoformat(),
            "commentary_provider": provider,
            "status": "complete"
        },
        "analytics_payload": [
            {
                "analysis_type": module,
                "chart_data": extract_chart_data(statistical_results["results"].get(module, {}), module),
                "model_comparison": extract_model_comparison(statistical_results["results"].get(module, {})),
                "ai_insights": {
                    "academic_commentary": commentary.get("academic_commentary", ""),
                    "business_translation": commentary.get("business_commentary", "")
                }
            }
            for module in config["project_metadata"]["selected_modules"]
            if module in statistical_results["results"]
        ],
        "summary": {
            "key_recommendations": commentary.get("key_recommendations", []),
            "confidence_level": commentary.get("confidence_level", "medium")
        }
    }
    
    # Save final payload
    save_final_payload(final_payload)
    
    # Update config
    config["run_info"]["status"] = "commented"
    save_config(config)
    
    print("\n" + "=" * 70)
    print("âœ… AI COMMENTARY COMPLETE")
    print("=" * 70)


def extract_chart_data(module_results: dict, module_name: str) -> dict:
    """Extract chart-ready data from module results."""
    if module_name == "descriptive":
        # Return first frequency distribution as sample
        freqs = module_results.get("frequencies", {})
        if freqs:
            first_var = list(freqs.keys())[0]
            return {
                "labels": freqs[first_var].get("labels", []),
                "values": freqs[first_var].get("counts", [])
            }
    
    elif module_name == "typology":
        return {
            "labels": [f"Cluster {i+1}" for i in range(module_results.get("optimal_k", 0))],
            "values": list(module_results.get("cluster_sizes", {}).values())
        }
    
    return {"labels": [], "values": []}


def extract_model_comparison(module_results: dict) -> dict:
    """Extract model comparison data if available."""
    if "winner" in module_results:
        return {
            "candidates": [c["model"] for c in module_results.get("candidates", [])],
            "winner": module_results.get("winner"),
            "metric": module_results.get("winner_metric"),
            "shap_summary": str(module_results.get("shap_summary", {}))
        }
    return {}


if __name__ == "__main__":
    main()

