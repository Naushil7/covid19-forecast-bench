import React, { Component } from "react";
import "./submit.css";
import "../forecastbench.css";
import "../w3.css";


class Submit extends Component {
    render() {
        return (

            <div class="w3-content">
            
                <div class = "w3-sand w3-sans-serif w3-large">
                
                    <h1>Submitting your Forecasts</h1>

                    <p>
                    We invite the interested teams to submit forecasts generated by their methodology by issuing a pull request.
                    </p>

                    <ul class="w3-ul">
                    <li> <b> Data to use</b>: While generating retrospective forecasts, please ensure that you are only using data that was available before then. 
                    Our Github repo provides the <a href="https://github.com/scc-usc/covid19-forecast-bench/tree/master/historical-data" target="_blank"> version of case and death data available on previous days </a>. 
                    The method should treat them as "separate datasets" without any knowledge of the date of the data, i.e., the method should be consistent irrespective of the date.

                    <div class ="w3-center">
                        <img src="https://raw.githubusercontent.com/scc-usc/covid19-forecast-bench/master/frontend/images/forecast-generation.png" height="300"/>
                    </div>
                    You may use the historical versions of the JHU data available at <a href=" https://github.com/scc-usc/ReCOVER-COVID-19/blob/master/results/historical_forecasts" target="_blank"> our other repo in time-series format</a>.
                    Other data sources may be used as long as you can guarantee that no "foresight" was used.
                    </li>
                    <li> <b> Submission format </b>:The format of the file should be exactly like the <a href="https://github.com/reichlab/covid19-forecast-hub/blob/master/data-processed/README.md#forecast-file-format" target="_blank"> submissions for Reich Lab's forecast hub</a>. Please follow the naming convention: [Date of forecast (YYYY-MM-DD)]-[Method_Name].csv. </li>
                    <li> <b> Forecast dates </b>: We will take the retrospective forecasts for any range of time, starting in July until the present. </li>
                    <li> <b> Forecast locations </b>: Currently, we are only accepting case and death forecasts for US state, county, and national-level and national-level for other countries. More locations will be addressed in the future. </li>
                    <li> <b> Forecast horizon</b>: The forecasts are expected to be incident cases forecasts per week observed on a Sunday for 1, 2, 3, and 4-week ahead. One week ahead forecast geenrated after a Monday is to be treated as the Sunday after the next one. This is in accordance with the Reich Lab's forecasting hub. </li>
                    <li> <b> Where to upload files </b>: Please add your files in the folder "raw-forecasts/" in your forked repo and submit a pull request. It can be done directly using your browser, cloning the repo is not needed:

                    <div class ="w3-center">
                        <img src="https://raw.githubusercontent.com/scc-usc/covid19-forecast-bench/master/frontend/images/pull-request.PNG" width="100%"/>
                    </div>
                    </li>                    
                    <li> <b> Methodology Description</b>: In a file named metadata-[Method Name].csv, please provide a short description of your approach, mentioning at least the following:
                    <ul>
                        <li> Modeling technique: Generative (SIR, SEIR, ...), Discriminative (Neural Networks, ARIMA, ...), ... </li>
                        <li> Learning approach: Bayesian, Regression (Quadratic Programming, Gradient Descent, ...), ... </li>
                        <li> Pre-processing: Smoothing (1 week, 2, week, auto), anomaly detection, ... </li>
                    </ul>
                    </li>
                    <li> <b>Submitting forecasts from multiple methodologies</b>: If you are submitting forecasts for multiple methodolgies, please ensure there is something in their metadata descriptions that differentiates them. Please note that any change in your approach, including data pre-processing and hand-tuning a parameter counts as a different methodology.
                    You can alter your method name to mark the distinction such as by appending an appropriate suffix. </li>
                    <li> <b>License</b>: We encourage you to make your submission under <a href="https://opensource.org/licenses/MIT" target="_blank"> the opensource MIT license</a>. </li>
                    </ul>
                    <br/>
                </div>
            </div>

        );
    }
}

export default Submit;