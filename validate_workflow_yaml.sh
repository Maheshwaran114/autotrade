#!/bin/bash
# YAML Syntax Validator for GitHub Actions Workflow
# This script checks for common YAML formatting errors in workflow files

set -e

# Define colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

WORKFLOW_FILE="$1"

if [ -z "$WORKFLOW_FILE" ]; then
  echo -e "${RED}Error: No workflow file specified.${NC}"
  echo "Usage: $0 <path-to-workflow-file.yml>"
  exit 1
fi

if [ ! -f "$WORKFLOW_FILE" ]; then
  echo -e "${RED}Error: File '$WORKFLOW_FILE' does not exist.${NC}"
  exit 1
fi

echo -e "${YELLOW}Validating YAML syntax in $WORKFLOW_FILE...${NC}"

# Check if yamllint is installed
if ! command -v yamllint &> /dev/null; then
  echo "Installing yamllint..."
  pip install yamllint || {
    echo -e "${RED}Failed to install yamllint. Falling back to basic checks.${NC}"
  }
fi

# If yamllint is available, use it for validation
if command -v yamllint &> /dev/null; then
  echo "Running yamllint validation..."
  yamllint -d relaxed "$WORKFLOW_FILE" && {
    echo -e "${GREEN}✅ YAML syntax is valid according to yamllint.${NC}"
  } || {
    echo -e "${RED}❌ YAML syntax errors detected by yamllint.${NC}"
    exit 1
  }
else
  # Basic validation for common issues if yamllint is not available
  echo "Performing basic YAML validation checks..."
  
  # Check for tabs
  if grep -P '\t' "$WORKFLOW_FILE" > /dev/null; then
    echo -e "${RED}❌ Tab characters detected. YAML requires spaces for indentation.${NC}"
    grep -n -P '\t' "$WORKFLOW_FILE"
    exit 1
  fi
  
  # Check for inconsistent indentation
  prev_indent=0
  line_num=0
  while IFS= read -r line; do
    ((line_num++))
    if [[ "$line" =~ ^[[:space:]]*$ || "$line" =~ ^[[:space:]]*# ]]; then
      continue # Skip empty lines and comments
    fi
    
    # Count leading spaces
    spaces=$(echo "$line" | sed -E 's/^( *).*/\1/' | wc -c)
    ((spaces--)) # Adjust for wc counting the newline
    
    # Check if indentation jumps by more than 2 spaces
    if (( spaces > prev_indent + 2 )) && (( prev_indent > 0 )); then
      echo -e "${RED}❌ Possible indentation error at line $line_num:${NC}"
      echo "   Previous indentation: $prev_indent spaces"
      echo "   Current indentation: $spaces spaces"
      echo "   Line content: $line"
    fi
    
    prev_indent=$spaces
  done < "$WORKFLOW_FILE"
  
  # Check for missing colons in mapping
  grep -n -E "^[[:space:]]*[^[:space:]:#-][^:]*$" "$WORKFLOW_FILE" > /dev/null && {
    echo -e "${RED}❌ Possible missing colon in mapping:${NC}"
    grep -n -E "^[[:space:]]*[^[:space:]:#-][^:]*$" "$WORKFLOW_FILE"
    exit 1
  } || echo -e "${GREEN}✅ No obvious missing colons detected.${NC}"
  
  # Check for here-documents (<<) that might cause issues
  HERE_DOCS=$(grep -n "<<" "$WORKFLOW_FILE" | grep -v "<<-")
  if [ -n "$HERE_DOCS" ]; then
    echo -e "${YELLOW}⚠️ Here-document (<<) found, which can cause issues if not properly indented:${NC}"
    echo "$HERE_DOCS"
    echo "Ensure proper indentation and quoting in heredocs."
  fi
fi

echo -e "${GREEN}✅ Basic YAML syntax validation completed.${NC}"

# Additional GitHub Actions specific checks
echo "Checking GitHub Actions specific patterns..."

# Check for ${{ }} syntax
if ! grep -q "\\${{" "$WORKFLOW_FILE"; then
  echo -e "${YELLOW}⚠️ No GitHub expression syntax (${{ }}) found. This might be expected, but unusual.${NC}"
fi

# Check for common runner OS
if ! grep -q "runs-on:" "$WORKFLOW_FILE"; then
  echo -e "${RED}❌ No 'runs-on' directive found. Jobs must specify a runner.${NC}"
  exit 1
fi

# Final success message
echo -e "${GREEN}✅ GitHub Actions workflow validation completed successfully!${NC}"
echo "Note: This is a basic check and may not catch all YAML syntax issues."
echo "Consider using GitHub's workflow validator through API or web interface for more thorough validation."
exit 0
