import numpy as np

from datetime import datetime

from .yaml_handler import YamlHandler as yamlh
from ..Deeplearn import setup, plot, fit



def ML(conf_file):
    yaml_handler = yamlh(conf_file)
     # Get current date -> %Y_%M-%D_%H.%M.%S
    date_now = str(datetime.now()).replace(' ',  '_')[:-7]
    yaml_handler.update_yaml('date', date_now)
    output_folder = f"{yaml_handler.read_key('savepath')}/output"

        # Load data
    print('Loading data...')
    train_generator, test_generator = setup.load_data(
        directory = f"{yaml_handler.read_key('savepath')}/output/imgs", 
        height = yaml_handler.read_key('height'), 
        lenght = yaml_handler.read_key('lenght')
        )

    # Build and compile model
    model =setup.build_and_compile_cnn_model(input_shape = train_generator.image_shape,
                                        n_classes = train_generator.num_classes
    )
    # Fit model
    print('Model fitting...')
    history = fit.fit_model(model, train_generator, test_generator)

    # Save model
    print('Saving model...')
    print(f"Saving model at {output_folder}/models")
    model.save(f'{output_folder}/models/{date_now}.h5')

    # Save  perforance plot
    print('Saving performance...')
    plot.get_performance(
        history= history,
        path_to_save=f'{output_folder}/performance/performance_{date_now}.svg')


    # Get predictions
    print(f'Getting predictions...')
    y_pred =  np.argmax(model.predict(test_generator), axis = 1)  


    # Save Confution Matrix
    print(f'Saving confution matrix...')
    plot.plot_confusion_matrix( 
        generator=test_generator,
        y_pred = y_pred,
        path_to_save= f'{output_folder}/performance/CM_{date_now}.svg'
     ) 
  



if __name__ == "__main__":
    ML()
    # yaml_handler = yamlh('config.yml')
    # print(yaml_handler.read_key('downsample_to'))