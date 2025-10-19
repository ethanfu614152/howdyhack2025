async function fetchUserData(userEmail){
    // In case the API is down, we want to use a try catch
    try{
        const response = await fetch("https://event-api.tamuhack.org/user", {
            method: "POST",
            headers: {
                Authorization: "Bearer tamuhackisbest"
            },
            body: JSON. stringify({
                email: userEmail
            })
        }
        )
        const data = await response.json()
        fillUserData(data)
        console.log(data)

    }
    catch(error)
    {

    }
}

function fillUserData(data){
    document.getElementById("name").innerText = data.first_name + " " + data.last_name
    document.getElementById("major").innerText = data.major
}
fetchUserData("tylertsai@tamu.edu")
