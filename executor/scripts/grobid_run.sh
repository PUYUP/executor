#!/bin/bash

# ============================================================================
# Interactive Menu Script for GROBID Docker
# ============================================================================
# Choose which GROBID variant to run
# ============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    GROBID Docker Runner                    ║"
echo "║           Choose GROBID variant to run                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📌 Recommendations:"
echo "   • Full Image + GPU   : Best accuracy, requires GPU"
echo "   • Full Image + CPU   : Best accuracy, slower"
echo "   • CRF-only Image     : Fast, small size"
echo ""
echo "Select option:"
echo "  1) Full Image with GPU"
echo "  2) Full Image with CPU"
echo "  3) CRF-only Image (Lightweight)"
echo "  4) Crossref Integration (Custom Config)"
echo "  5) Custom: Enter manual docker run command"
echo "  0) Exit"
echo ""
read -p "Enter choice (0-5): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting Full Image with GPU..."
        GROBID_VERSION="0.9.0"
        docker pull "grobid/grobid:${GROBID_VERSION}-full"
        docker run --rm \
            --gpus all \
            --init \
            --ulimit core=0 \
            -p 8070:8070 \
            "grobid/grobid:${GROBID_VERSION}-full"
        ;;
    2)
        echo ""
        echo "🚀 Starting Full Image with CPU..."
        GROBID_VERSION="0.9.0"
        docker pull "grobid/grobid:${GROBID_VERSION}-full"
        docker run --rm \
            --init \
            --ulimit core=0 \
            -p 8070:8070 \
            "grobid/grobid:${GROBID_VERSION}-full"
        ;;
    3)
        echo ""
        echo "🚀 Starting CRF-only Image..."
        GROBID_VERSION="0.9.0"
        docker pull "grobid/grobid:${GROBID_VERSION}-crf"
        docker run --rm \
            --init \
            --ulimit core=0 \
            -p 8070:8070 \
            "grobid/grobid:${GROBID_VERSION}-crf"
        ;;
    4)
        echo ""
        echo "🚀 Starting Crossref Integration Setup..."
        if command -v bash &> /dev/null; then
            # Try to run setup_crossref.sh if it exists
            SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
            if [ -f "$SCRIPT_DIR/setup_crossref.sh" ]; then
                bash "$SCRIPT_DIR/setup_crossref.sh"
            else
                echo "setup_crossref.sh not found. Please ensure it's in the same directory."
                echo "You can manually run: bash setup_crossref.sh"
            fi
        fi
        ;;
    5)
        echo ""
        echo "📝 Custom Mode: Enter docker run command"
        echo "Example: docker run --rm --gpus all --init --ulimit core=0 -p 8080:8070 grobid/grobid:0.9.0-full"
        echo ""
        read -p "Enter full command: " custom_cmd
        
        # Run custom command
        eval "$custom_cmd"
        ;;
    0)
        echo "👋 Exiting program"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please try again."
        exit 1
        ;;
esac

echo ""
echo "✅ Container has stopped"