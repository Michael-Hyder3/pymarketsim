#!/bin/bash
# Run comparison test until we find a >100% case

cd "/Users/mkh134/Desktop/Fall 25 Research/pymarketsim"

# Activate virtual environment
source .venv/bin/activate

for i in {1..20}; do
    echo "=============== Run $i ==============="
    python -m marketsim.tests.comparison_test 2>&1 > /tmp/run_$i.txt
    
    # Check if we found >100%
    if grep -q "DETECTED >100%" /tmp/run_$i.txt; then
        echo "Found >100% case in run $i!"
        cat /tmp/run_$i.txt
        exit 0
    fi
    
    # Show percentage
    pct=$(grep "% of Sorted-Optimal" /tmp/run_$i.txt | tail -1 | awk '{print $6}')
    echo "Run $i: HBL = $pct%"
done

echo "No >100% cases found in 20 runs"
