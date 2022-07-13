






# Notification received of the Bitcoin predition price for today
def notification(price):
    title = 'BTC Prediction Tracker',
    message = 'Todays prediction is!', price
    app_icon = None,
    timeout = 10
    ticker = 'Todays prediction is!'

toaster = win10toast.ToastNotifier()

toaster.show_toast("BTC Prediction Tracker", "Today's prediction is!", duration=10) 