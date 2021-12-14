# Python Librariesimport urllibimport requestsimport jsonfrom datetime import datetime# Local modulesfrom config import user, pswd# Format access pointURL = 'https://st-dev-data-api.azurewebsites.net'# url = urllib.parse.urljoin(URL, '/api/v0.1/buildings/energyville1/transfo/realtime/properties')consumption_url = '/api/v0.1/buildings/energyville1/transfo/realtime'weather_url = '/api/v0.1/weather/forecasts/darksky/hourly/latest/'url = urllib.parse.urljoin(URL, '/api/v0.1/buildings/energyville1/transfo/realtime')# Format parametersDATE_FORMAT_STR = "%Y-%m-%d"start = datetime(2021, 11, 1)end = datetime(2021, 11, 2)params = {    # SmarThor parameters    'start': start.strftime(DATE_FORMAT_STR),    'end': end.strftime(DATE_FORMAT_STR),    'time_zone': 'Central European Standard Time',    # Endpoint specific parameters    'transfo_id': 'transfo1',    'properties': 'ActiveImport'}if __name__ == "__main__":    # Send request    result = requests.get(url, params=params, auth=(user, pswd))    # Visualise / Save response    print(result.status_code)    # print(result.text)    if result.ok:        measurements, time = zip(*[(reading['ActiveImport'], reading['DateTimeMeasurement'])                                   for reading in result.json()['data']])        with open('data.json', 'w') as outfile:            json.dump({'time': time, 'values': measurements}, outfile)