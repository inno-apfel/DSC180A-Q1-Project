from threading import Thread
from threading import Event
import time

class AnimationLoader():
    """
    Asynchronous loader for displaying loading animation with messages.
    """

    def __init__(self):
        """
        Initializes AnimationLoader with default values.
        """
        self.message = ""
        self.finish_message = ""
        self.__failed = False
        self.__finished = False
        self.failed_message = ""
        self.__threadEvent = Event()
        self.__thread = Thread(target=self.__loading, daemon=True)
        self.__threadBlockEvent = Event()

    @property
    def finished(self):
        """
        bool: Indicates whether the loading process has finished successfully.
        """
        return self.__finished

    @finished.setter
    def finished(self, finished):
        """
        Sets the 'finished' attribute.

        Parameters
        ----------
        finished : bool
            Whether or not the loading process has finished successfully.
        
        Raises
        ------
        ValueError
            If the input is not a boolean.
        """
        if isinstance(finished, bool):
            self.__finished = finished
            if finished:
                self.__threadEvent.set()
                time.sleep(0.1)
        else:
            raise ValueError

    @property
    def failed(self):
        """
        bool: Indicates whether the loading process has failed.
        """
        return self.__failed

    @failed.setter
    def failed(self, failed):
        """
        Sets the 'failed' attribute.

        Parameters
        ----------
        failed : bool
            Whether or not the loading process has failed.
        
        Raises
        ------
        ValueError
            If the input is not a boolean.
        """
        if isinstance(failed, bool):
            self.__failed = failed
            if failed:
                self.__threadEvent.set()
                time.sleep(0.1)
        else:
            raise ValueError

    def show(self, loading_message: str, finish_message: str = '✅ Finished', failed_message='❌ Failed'):
        """
        Sets loading messages and initiates the loading animation.

        Parameters
        ----------
        loading_message : str
            Message displayed during loading.
        
        finish_message : str, optional
            Message displayed upon successful completion.

        failed_message : str, optional
            Message displayed upon failure.
        """
        self.message = loading_message
        self.finish_message = finish_message
        self.failed_message = failed_message
        self.show_loading()

    def show_loading(self):
        """
        Starts the loading animation.
        """
        self.finished = False
        self.failed = False
        self.__threadEvent.clear()
        if not self.__thread.is_alive():
            self.__thread.start()
        else:
            self.__threadBlockEvent.set()

    def __loading(self):
        """
        Method containing animation code to be executed
        """
        symbols = ['/', '-', '\\', '|']
        i = 0
        while True:
            # print('')
            while not self.finished and not self.failed:
                i = (i + 1) % len(symbols)
                print('\r\033[K%s %s' % (self.message, symbols[i]), flush=True, end='')
                self.__threadEvent.wait(0.1)
                self.__threadEvent.clear()
            if self.finished is True and not self.failed:
                print('\r\033[K%s' % self.finish_message, flush=True)
            else:
                print('\r\033[K%s' % self.failed_message, flush=True)
            # print('')
            self.__threadBlockEvent.wait()
            self.__threadBlockEvent.clear()
