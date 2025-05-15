#!/bin/bash
# GitHub Actions Workflow SSH Validator
# This script specifically checks for common SSH and heredoc issues in GitHub Actions workflows

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
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

echo -e "${BLUE}Validating GitHub Actions workflow SSH commands in $WORKFLOW_FILE...${NC}"

# Create a directory for temporary files
TMP_DIR="./workflow_validation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TMP_DIR"

# Function to extract and validate SSH heredoc sections
validate_ssh_heredocs() {
  local file="$1"
  local output_file="$TMP_DIR/heredoc_sections.txt"
  
  echo "Extracting and validating SSH heredoc sections..."
  
  # Extract potential SSH sections with heredocs
  grep -n -A 20 "ssh " "$file" > "$output_file"
  
  # Check for common issues in SSH heredoc sections
  if grep -q "<<[^-]EOF" "$output_file"; then
    echo -e "${YELLOW}⚠️ Found standard heredoc (<<EOF) which might cause indentation issues${NC}"
    echo "Consider using <<-EOF for indented heredocs in YAML"
    grep -n "<<[^-]EOF" "$file"
  fi
  
  # Check for EOF tokens not at the start of a line
  if grep -q "^[[:space:]][[:space:]]*EOF" "$output_file"; then
    echo -e "${RED}❌ Found indented EOF token, which should be at the start of a line${NC}"
    echo "The closing EOF must not be indented in standard heredocs (<<EOF)"
    grep -n -A 1 "<<EOF" "$file"
  fi
  
  # Check for UserKnownHostsFile with /null instead of /dev/null
  if grep -q "UserKnownHostsFile /null" "$file"; then
    echo -e "${RED}❌ Incorrect UserKnownHostsFile path: '/null' should be '/dev/null'${NC}"
    grep -n "UserKnownHostsFile /null" "$file"
  fi
  
  return 0
}

# Function to check for correct SSH options
validate_ssh_options() {
  local file="$1"
  
  echo "Validating SSH options..."
  
  # Check for missing StrictHostKeyChecking option
  if grep -q "ssh " "$file" && ! grep -q "StrictHostKeyChecking" "$file"; then
    echo -e "${YELLOW}⚠️ SSH commands found, but no StrictHostKeyChecking option${NC}"
    echo "Consider adding -o StrictHostKeyChecking=no to avoid prompts"
  fi
  
  # Check for missing connection timeout
  if grep -q "ssh " "$file" && ! grep -q "ConnectTimeout" "$file"; then
    echo -e "${YELLOW}⚠️ SSH commands found, but no ConnectTimeout option${NC}"
    echo "Consider adding -o ConnectTimeout=<seconds> to avoid hanging on connection issues"
  fi
  
  # Check for proper IdentityFile format
  if grep -q "IdentityFile" "$file"; then
    if ! grep -q "IdentityFile.*~/.ssh/" "$file"; then
      echo -e "${YELLOW}⚠️ IdentityFile path might be incorrect${NC}"
      echo "SSH keys are typically stored in ~/.ssh/ directory"
      grep -n "IdentityFile" "$file"
    fi
  fi
  
  return 0
}

# Function to validate general YAML structure
validate_yaml_structure() {
  local file="$1"
  
  echo "Validating YAML structure..."
  
  # Check for common YAML syntax errors
  
  # Check for tabs
  if grep -P '\t' "$file" > /dev/null; then
    echo -e "${RED}❌ Tab characters detected. YAML requires spaces for indentation.${NC}"
    grep -n -P '\t' "$file"
    return 1
  fi
  
  # Check for colons without spaces
  if grep -P ':[^ \n#]' "$file" | grep -v "://" > /dev/null; then
    echo -e "${YELLOW}⚠️ Found colons without spaces after them, which might cause issues:${NC}"
    grep -n -P ':[^ \n#]' "$file" | grep -v "://"
  fi
  
  # Check for misaligned colons
  prev_colon_pos=0
  line_num=0
  while IFS= read -r line; do
    ((line_num++))
    if [[ "$line" =~ ^([[:space:]]*)[^[:space:]#-]*: ]]; then
      indent=${#BASH_REMATCH[1]}
      # Extract position of the first colon
      colon_pos=$(echo "$line" | sed -E 's/^( *)[^ ]*(:).*/\1\2/' | wc -c)
      
      # If we have a previous indent at the same level but different colon position
      if [[ $indent -eq $prev_indent && $colon_pos -ne $prev_colon_pos && $prev_colon_pos -ne 0 ]]; then
        echo -e "${YELLOW}⚠️ Possible misaligned colons at line $line_num:${NC}"
        echo "   Previous colon position: $prev_colon_pos"
        echo "   Current colon position: $colon_pos"
        echo "   Line content: $line"
      fi
      
      prev_indent=$indent
      prev_colon_pos=$colon_pos
    fi
  done < "$file"
  
  return 0
}

# Validate GitHub Actions workflow
validate_github_actions_workflow() {
  local file="$1"
  
  echo "Validating GitHub Actions workflow structure..."
  
  # Check for missing required sections
  if ! grep -q "^name:" "$file"; then
    echo -e "${RED}❌ Missing 'name' field in workflow${NC}"
  fi
  
  if ! grep -q "^on:" "$file"; then
    echo -e "${RED}❌ Missing 'on' trigger in workflow${NC}"
  fi
  
  if ! grep -q "^jobs:" "$file"; then
    echo -e "${RED}❌ Missing 'jobs' section in workflow${NC}"
  fi
  
  # Check for common workflow issues
  if grep -q "uses:" "$file" && grep -q "run:" "$file"; then
    # Check for actions using different @ version formats
    if grep -q "uses:.*@v[0-9]" "$file" && grep -q "uses:.*@[^v][0-9]" "$file"; then
      echo -e "${YELLOW}⚠️ Inconsistent version formats in 'uses' directives${NC}"
      echo "Some actions use @v2 format while others use @2 format"
      grep -n "uses:.*@" "$file"
    fi
    
    # Check for run commands without proper pipefail
    RUN_COMMANDS=$(grep -n "run:" "$file" | grep -v "set -e" | grep -v "set -o pipefail")
    if [ -n "$RUN_COMMANDS" ]; then
      echo -e "${YELLOW}⚠️ Some 'run' commands don't set failure options${NC}"
      echo "Consider adding 'set -e' or 'set -o pipefail' to ensure errors are caught"
      echo "$RUN_COMMANDS" | head -5
    fi
  fi
  
  return 0
}

# Run all validation functions
validate_yaml_structure "$WORKFLOW_FILE"
validate_ssh_heredocs "$WORKFLOW_FILE"
validate_ssh_options "$WORKFLOW_FILE"
validate_github_actions_workflow "$WORKFLOW_FILE"

# Final check for actual YAML parsing
echo -e "${BLUE}Performing final YAML syntax check...${NC}"

# Check if yamllint is installed
if command -v yamllint &> /dev/null; then
  yamllint -d relaxed "$WORKFLOW_FILE" && {
    echo -e "${GREEN}✅ YAML syntax is valid according to yamllint${NC}"
  } || {
    echo -e "${RED}❌ YAML syntax errors detected by yamllint${NC}"
    exit 1
  }
else
  # Alternative check using python if available
  if command -v python3 &> /dev/null; then
    python3 -c "import yaml; yaml.safe_load(open('$WORKFLOW_FILE'))" 2> "$TMP_DIR/yaml_errors.txt"
    if [ $? -eq 0 ]; then
      echo -e "${GREEN}✅ YAML syntax is valid according to Python yaml parser${NC}"
    else
      echo -e "${RED}❌ YAML syntax errors detected by Python yaml parser:${NC}"
      cat "$TMP_DIR/yaml_errors.txt"
      exit 1
    fi
  else
    echo -e "${YELLOW}⚠️ No YAML parser available for final validation${NC}"
  fi
fi

echo -e "${GREEN}✅ SSH command validation in workflow completed successfully!${NC}"
echo "Remember to manually review SSH commands and heredocs for proper formatting"
echo "Temporary files saved in: $TMP_DIR"

exit 0
