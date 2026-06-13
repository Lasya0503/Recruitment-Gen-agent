import pydantic

class TenantProfile(pydantic.BaseModel):
    company_name: str
    department: str      # 🟢 FIXED: Swapped 'target_domain' to 'department'
    scheduling_url: str

# Default fallback structural initialization profiles
DEFAULT_TENANT = TenantProfile(
    company_name="TechCorp Solutions",
    department="Backend Engineering Infrastructure", # 🟢 FIXED: Swapped keyword parameter
    scheduling_url="https://calendly.com/techcorp-evaluation-panel"
)