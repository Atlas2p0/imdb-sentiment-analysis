from src.preprocessing import preprocess_text_for_lstm, preprocess_text_for_rnn, preprocess_text_for_bert
from src.utils import load_vocab, predict_lstm, predict_rnn, predict_bert, load_tokenizer
import src.config as cfg
import os
import colorama
from colorama import Fore, Style
import pyfiglet


colorama.init(autoreset=True)

def print_title():
    """Generates and prints the ASCII art title."""
    # Choose a font (check pyfiglet fonts online, 'slant' is a common one)
    # The title is long, might need a smaller font or break lines
    title_part1 = "Deep Learning"
    title_part2 = "Sentiment Analysis"
    try:
        fig = pyfiglet.Figlet(font='slant') # Or 'standard', 'big', 'banner'
        # Print with a distinct color
        print(Fore.CYAN + Style.BRIGHT + fig.renderText(title_part1) + Style.RESET_ALL)
        print(Fore.CYAN + Style.BRIGHT + fig.renderText(title_part2) + Style.RESET_ALL)
    except Exception as e:
        # Fallback if pyfiglet fails
        print(Fore.CYAN + Style.BRIGHT + "===== Deep Learning Sentiment Analysis =====" + Style.RESET_ALL)
        print(f"ASCII art failed: {e}")

def styled_print(text, color, style=''):
    """Prints text with specified color and style."""
    print(color + style + str(text) + Style.RESET_ALL)

def read_multiline_input(prompt):
    """Reads multiple lines from the terminal until an empty line is entered."""
    styled_print("Enter your review. Press Enter and then '!EOI' to finish your input.", Fore.RED)

    lines= []
    line= input()
    lines.append(line)
    while  line!= '!EOI':
        line= input()
        lines.append(line)
    formatted_input= "\n".join(lines).replace('\n', ' ')
    return formatted_input

if __name__ == '__main__':
    option= 0
    sentiment= ''
    print_title()
    
    while option != 4:
        
        styled_print("Please choose model for sentiment prediction", color= Fore.GREEN)
        styled_print("1: RNN", color= Fore.GREEN)
        styled_print("2: LSTM", color= Fore.GREEN)
        styled_print("3: BERT", color= Fore.GREEN)
        styled_print("4: Exit", color= Fore.GREEN)

        user_input= input(Fore.CYAN + "Enter Option: " + Style.RESET_ALL).replace('\n', ' ')
            

        try:
            option= int(user_input)
        except ValueError:
            option= -1
    
        match option:
            case 1:
                #RNN Inference
                review= read_multiline_input("Please enter the review for prediction: ")
                rnn_vocab_path= os.getcwd() + "/artifacts/rnn_vocab.pkl"
                rnn_vocab= load_vocab(rnn_vocab_path)
                rnn_model_file_path= os.getcwd() + "/artifacts/parameter_tuned_rnn.pth"
                index_tensor= preprocess_text_for_rnn(review, vocab= rnn_vocab, max_len= cfg.RNN_VOCAB_SIZE)
                sentiment= predict_rnn(index_tensor, rnn_model_file_path)
                styled_print(f"This movie review is {sentiment} ", Fore.LIGHTRED_EX)
            case 2:
                # LSTM Inference 
                review= read_multiline_input("Please enter the review for prediction: ")
                lstm_vocab_path= os.getcwd() + "/artifacts/lstm_vocab.pkl"    
                lstm_model_file_path= os.getcwd() + "/artifacts/simple_lstm.pth"
                lstm_vocab= load_vocab(lstm_vocab_path)
                index_tensor, length_tensor= preprocess_text_for_lstm(review, vocab= lstm_vocab, max_len= cfg.MAX_LENGTH_LSTM)
                sentiment= predict_lstm(index_tensor, length_tensor, lstm_model_file_path)
                styled_print(f"This movie review is {sentiment}", Fore.LIGHTRED_EX)
            case 3:
                # BERT Inference
                review= read_multiline_input("Please enter the review for prediction: ")
                bert_tokenizer_path= os.getcwd() + "/artifacts/bert_tokenizer/"
                bert_model_path= os.getcwd() + "/artifacts/bert_1_linear_layer.pth"
                loaded_bert_tokenizer= load_tokenizer(bert_tokenizer_path)
                input_ids, attention_mask, token_type_ids= preprocess_text_for_bert(raw_text= review, tokenizer= loaded_bert_tokenizer, max_len= cfg.MAX_LENGTH_BERT)
                sentiment= predict_bert(input_ids= input_ids.unsqueeze(0),
                                        attention_mask= attention_mask.unsqueeze(0),
                                        token_type_ids= token_type_ids.unsqueeze(0),
                                        model_file_path= bert_model_path)
                styled_print(f"This movie review is: {sentiment} ", Fore.LIGHTRED_EX)
            case 4:
                print("Exiting...")
            case _:
                print("Invalid option entered, please try again...")



