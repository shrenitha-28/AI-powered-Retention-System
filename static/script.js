// ================================
// Customer Churn Prediction
// script.js
// ================================

// Inputs
const internet = document.getElementById("InternetService");
const onlineSecurity = document.getElementById("OnlineSecurity");
const onlineBackup = document.getElementById("OnlineBackup");
const deviceProtection = document.getElementById("DeviceProtection");
const techSupport = document.getElementById("TechSupport");
const streamingTV = document.getElementById("StreamingTV");
const streamingMovies = document.getElementById("StreamingMovies");

const form = document.getElementById("predictionForm");
const predictBtn = document.getElementById("predictBtn");

const quickButtons = document.querySelectorAll(".quick-buttons button");

// =====================================
// Automatically update services
// =====================================

function updateInternetServices(){

    const services = [
        onlineSecurity,
        onlineBackup,
        deviceProtection,
        techSupport,
        streamingTV,
        streamingMovies
    ];

    if(internet.value === "No"){

        services.forEach(service=>{

            service.innerHTML =
                `<option value="No internet service">
                    No internet service
                </option>`;

            service.value = "No internet service";

            service.disabled = true;

        });

    }
    else{

        services.forEach(service=>{

            service.disabled = false;

            service.innerHTML = `
                <option value="No">No</option>
                <option value="Yes">Yes</option>
            `;

        });

    }

}

internet.addEventListener("change",updateInternetServices);

updateInternetServices();

// =====================================
// Hidden Encoding
// =====================================

form.addEventListener("submit",function(){

    // Multiple Lines
    document.getElementById("MultipleLines_Yes").value =
        document.getElementById("MultipleLines").value;

    // Internet Service
    document.getElementById("InternetService_Fiber_optic").value =
        internet.value==="Fiber Optic" ? 1 : 0;

    document.getElementById("InternetService_No").value =
        internet.value==="No" ? 1 : 0;

    // Online Security
    document.getElementById("OnlineSecurity_No_internet_service").value =
        onlineSecurity.value==="No internet service"?1:0;

    document.getElementById("OnlineSecurity_Yes").value =
        onlineSecurity.value==="Yes"?1:0;

    // Online Backup
    document.getElementById("OnlineBackup_No_internet_service").value =
        onlineBackup.value==="No internet service"?1:0;

    document.getElementById("OnlineBackup_Yes").value =
        onlineBackup.value==="Yes"?1:0;

    // Device Protection
    document.getElementById("DeviceProtection_No_internet_service").value =
        deviceProtection.value==="No internet service"?1:0;

    document.getElementById("DeviceProtection_Yes").value =
        deviceProtection.value==="Yes"?1:0;

    // Tech Support
    document.getElementById("TechSupport_No_internet_service").value =
        techSupport.value==="No internet service"?1:0;

    document.getElementById("TechSupport_Yes").value =
        techSupport.value==="Yes"?1:0;

    // Streaming TV
    document.getElementById("StreamingTV_No_internet_service").value =
        streamingTV.value==="No internet service"?1:0;

    document.getElementById("StreamingTV_Yes").value =
        streamingTV.value==="Yes"?1:0;

    // Streaming Movies
    document.getElementById("StreamingMovies_No_internet_service").value =
        streamingMovies.value==="No internet service"?1:0;

    document.getElementById("StreamingMovies_Yes").value =
        streamingMovies.value==="Yes"?1:0;

    // Payment Method
    const payment=document.getElementById("PaymentMethod").value;

    document.getElementById("PaymentMethod_Credit_card").value =
        payment==="Credit Card"?1:0;

    document.getElementById("PaymentMethod_Electronic_check").value =
        payment==="Electronic Check"?1:0;

    document.getElementById("PaymentMethod_Mailed_check").value =
        payment==="Mailed Check"?1:0;

    // SIM

    const sim=document.getElementById("SIM").value;

    document.getElementById("SIM_BSNL").value =
        sim==="BSNL"?1:0;

    document.getElementById("SIM_Jio").value =
        sim==="Jio"?1:0;

    document.getElementById("SIM_Vodafone").value =
        sim==="Vodafone"?1:0;

    // Contract

    const contract=document.getElementById("Contract").value;

    if(contract==="Month-to-month")
        document.getElementById("Contract_res").value=0;

    else if(contract==="One year")
        document.getElementById("Contract_res").value=1;

    else
        document.getElementById("Contract_res").value=2;

    // Button Animation

    predictBtn.innerHTML=
        '<i class="fa-solid fa-spinner fa-spin"></i> Predicting...';

});
