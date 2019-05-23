import unittest
import googleapiclient

class MyTestCase(unittest.TestCase):
    def test_something(self):
        model = 'black_friday'
        project = 'gad-playground-212407'
        version = 'v1'

        service = googleapiclient.discovery.build('ml', 'v1')
        name = 'projects/{}/models/{}'.format(project, model)

        if version is not None:
            name += '/versions/{}'.format('v1')

        response = service.projects().predict(
            name=name,
            body={'instances': [[0.0, 670.0, 0.0, 0.0, 10.0, 0.0, 2.0, 0.0, 2.0, None, None],
                                [0.0, 2374.0, 0.0, 0.0, 10.0, 0.0, 2.0, 0.0, 0.0, 4.0, 10.0],
                                [0.0, 850.0, 0.0, 0.0, 10.0, 0.0, 2.0, 0.0, 11.0, None, None]]}
        ).execute()

        if 'error' in response:
            raise RuntimeError(response['error'])

        response['predictions']

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
