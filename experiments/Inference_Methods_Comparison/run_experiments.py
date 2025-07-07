#!/usr/bin/env python3
"""
Simple launcher script for inference comparison experiments.
Provides easy access to different experiment modes.
"""

import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    print("🚀 Inference Methods Comparison Launcher")
    print("=" * 50)
    print("Compare different transformer inference methods:")
    print("• Standard generation")
    print("• Speculative decoding") 
    print("• Beam search")
    print("• Batched generation")
    print("• Nucleus sampling")
    print("=" * 50)

def show_menu():
    """Show main menu options"""
    print("\n📋 Available Options:")
    print("1. 🧪 Test setup (check dependencies)")
    print("2. ⚡ Quick experiment (minimal settings)")
    print("3. 🔧 Interactive configuration")
    print("4. 📄 Use configuration file")
    print("5. 📚 View documentation")
    print("6. 🚪 Exit")

def run_command(cmd):
    """Run a command and handle errors"""
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return False

def test_setup():
    """Run setup test"""
    print("\n🧪 Running setup test...")
    return run_command("python test_setup.py")

def quick_experiment():
    """Run quick experiment"""
    print("\n⚡ Running quick experiment...")
    print("This will test all methods with minimal settings (2 completions, 50 tokens)")
    confirm = input("Continue? [Y/n]: ").strip().lower()
    if confirm not in ['', 'y', 'yes']:
        return True
    
    return run_command("python inference_comparison.py --quick")

def interactive_experiment():
    """Run interactive experiment"""
    print("\n🔧 Starting interactive configuration...")
    return run_command("python inference_comparison.py --interactive")

def config_file_experiment():
    """Run experiment with config file"""
    print("\n📄 Available configuration files:")
    config_files = list(Path(".").glob("*.json"))
    
    if not config_files:
        print("No configuration files found in current directory.")
        print("You can create one based on config_example.json")
        return True
    
    for i, config_file in enumerate(config_files, 1):
        print(f"  {i}. {config_file}")
    
    try:
        choice = input(f"\nSelect config file (1-{len(config_files)}) or path: ").strip()
        
        if choice.isdigit():
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(config_files):
                config_path = config_files[choice_idx]
            else:
                print("Invalid choice")
                return True
        else:
            config_path = Path(choice)
            if not config_path.exists():
                print(f"File not found: {config_path}")
                return True
        
        print(f"Using config file: {config_path}")
        return run_command(f"python inference_comparison.py --config {config_path}")
        
    except ValueError:
        print("Invalid input")
        return True

def view_documentation():
    """View documentation"""
    print("\n📚 Documentation:")
    print("-" * 30)
    
    readme_path = Path("README.md")
    if readme_path.exists():
        with open(readme_path, 'r') as f:
            content = f.read()
        print(content)
    else:
        print("README.md not found")
        print("\nBasic usage:")
        print("python inference_comparison.py --help")
    
    input("\nPress Enter to continue...")
    return True

def main():
    """Main launcher function"""
    print_banner()
    
    while True:
        show_menu()
        
        try:
            choice = input("\n🎯 Select option (1-6): ").strip()
            
            if choice == '1':
                test_setup()
            elif choice == '2':
                quick_experiment()
            elif choice == '3':
                interactive_experiment()
            elif choice == '4':
                config_file_experiment()
            elif choice == '5':
                view_documentation()
            elif choice == '6':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-6.")
                continue
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue
        
        # Ask if user wants to continue
        if choice in ['1', '2', '3', '4']:
            continue_choice = input("\n🔄 Run another experiment? [Y/n]: ").strip().lower()
            if continue_choice in ['n', 'no']:
                print("\n👋 Goodbye!")
                break

if __name__ == "__main__":
    main()
