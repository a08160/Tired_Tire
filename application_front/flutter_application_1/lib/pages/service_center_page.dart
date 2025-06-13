import 'dart:convert';
import 'dart:html' as html;
import 'dart:js' as js;
import 'dart:ui_web' as ui;
import 'dart:math';
import 'package:csv/csv.dart';
import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';

class ServiceCenterPage extends StatefulWidget {
  const ServiceCenterPage({Key? key}) : super(key: key);

  @override
  _ServiceCenterPageState createState() => _ServiceCenterPageState();
}

class _ServiceCenterPageState extends State<ServiceCenterPage> {
  List<ServiceCenter> centers = [];
  List<ServiceCenter> nearbyCenters = [];
  double? userLat;
  double? userLng;

  @override
  void initState() {
    super.initState();
    _loadCsv();
    if (kIsWeb) {
      _registerMapViewFactory();
    }
  }

  Future<void> _loadCsv() async {
    final rawData = await rootBundle.loadString('assets/service_center.csv');
    final rows = const CsvToListConverter(eol: '\n').convert(rawData);

    List<ServiceCenter> parsedCenters = [];
    for (int i = 1; i < rows.length; i++) {
      final row = rows[i];
      double? lat = _toDouble(row[4]);
      double? lng = _toDouble(row[5]);
      if (lat != null && lng != null) {
        parsedCenters.add(
          ServiceCenter(
            name: row[0].toString(),
            lat: lat,
            lng: lng,
            address: row[2].toString(),
            phone: row[14].toString(),
          ),
        );
      }
    }

    setState(() {
      centers = parsedCenters;
    });
  }

  double? _toDouble(dynamic value) {
    if (value is double) return value;
    if (value is int) return value.toDouble();
    if (value is String) return double.tryParse(value);
    return null;
  }

  double calculateDistance(lat1, lon1, lat2, lon2) {
    const p = 0.017453292519943295;
    final a =
        0.5 -
        cos((lat2 - lat1) * p) / 2 +
        cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2;
    return 12742 * asin(sqrt(a));
  }

  void _registerMapViewFactory() {
    ui.platformViewRegistry.registerViewFactory('kakao-map', (int viewId) {
      final mapDiv =
          html.DivElement()
            ..id = 'kakao-map'
            ..style.width = '100%'
            ..style.height = '300px';

      const apiKey = '74c7746b63844e7a43e0c7f534ffcf40';

      final kakaoJs = """
        kakao.maps.load(function() {
          var container = document.getElementById('kakao-map');
          var options = {
            center: new kakao.maps.LatLng(37.5665, 126.9780),
            level: 3
          };
          window.map = new kakao.maps.Map(container, options);

          if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(function(position) {
              var lat = position.coords.latitude;
              var lon = position.coords.longitude;

              var locPosition = new kakao.maps.LatLng(lat, lon);
              window.map.setCenter(locPosition);

              var userMarker = new kakao.maps.Marker({
                map: window.map,
                position: locPosition,
                title: "내 위치"
              });

              var infowindow = new kakao.maps.InfoWindow({
                content: '<div style="padding:5px;font-size:12px;">현재 위치</div>'
              });
              infowindow.open(window.map, userMarker);

              window.dispatchEvent(new CustomEvent('user-location', {
                detail: { lat: lat, lng: lon }
              }));

            }, function(error) {
              console.error('위치 정보 가져오기 실패:', error);
            });
          }
        });
      """;

      if (html.document.getElementById('kakao-map-script') == null) {
        final script =
            html.ScriptElement()
              ..id = 'kakao-map-script'
              ..type = 'text/javascript'
              ..src =
                  'https://dapi.kakao.com/v2/maps/sdk.js?appkey=$apiKey&autoload=false&libraries=services'
              ..onLoad.listen((event) {
                js.context.callMethod('eval', [kakaoJs]);
              });

        html.document.body!.append(script);
      } else {
        js.context.callMethod('eval', [kakaoJs]);
      }

      html.window.addEventListener('user-location', (event) {
        final detail = (event as html.CustomEvent).detail;
        if (detail != null) {
          final lat = detail['lat'];
          final lng = detail['lng'];
          _onUserLocation(lat, lng);
        }
      });

      return mapDiv;
    });
  }

  void _onUserLocation(double lat, double lng) {
    setState(() {
      userLat = lat;
      userLng = lng;
    });

    final nearby =
        centers.map((center) {
            final dist = calculateDistance(lat, lng, center.lat, center.lng);
            return MapEntry(center, dist);
          }).toList()
          ..sort((a, b) => a.value.compareTo(b.value));

    final top5 = nearby.take(5).map((e) => e.key).toList();

    setState(() {
      nearbyCenters = top5;
    });

    _showMarkersOnMap(top5);
  }

  void _showMarkersOnMap(List<ServiceCenter> centers) {
    final markersJs =
        centers.map((c) {
          final content = "${c.name}<br>${c.address}";
          return """
        var marker = new kakao.maps.Marker({
          map: window.map,
          position: new kakao.maps.LatLng(${c.lat}, ${c.lng}),
          title: "${c.name}"
        });
        var infowindow = new kakao.maps.InfoWindow({
          content: '<div style="padding:5px;font-size:12px;">$content</div>'
        });
        kakao.maps.event.addListener(marker, 'click', function() {
          infowindow.open(window.map, marker);
        });
      """;
        }).join();

    js.context.callMethod('eval', [markersJs]);
  }

  @override
  Widget build(BuildContext context) {
    if (!kIsWeb) {
      return Scaffold(
        appBar: AppBar(title: const Text('정비소 찾기')),
        backgroundColor: Colors.white,
        body: const Center(child: Text('웹 환경에서만 지원됩니다.')),
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text('정비소 찾기')),
      backgroundColor: Colors.white,
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(
                height: 300,
                child: HtmlElementView(viewType: 'kakao-map'),
              ),
              const SizedBox(height: 30),
              ListView.builder(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: nearbyCenters.length,
                itemBuilder: (context, index) {
                  final center = nearbyCenters[index];
                  return Card(
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.zero,
                    ),
                    elevation: 0,
                    color: Colors.white,
                    margin: const EdgeInsets.symmetric(vertical: 8),
                    child: ListTile(
                      leading: CircleAvatar(
                        backgroundColor: Colors.grey.shade300,
                        child: Text(
                          '${index + 1}',
                          style: const TextStyle(color: Colors.white),
                        ),
                      ),
                      title: Text(
                        center.name,
                        style: const TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                      subtitle: Text(center.address),
                      isThreeLine: false,
                    ),
                  );
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class ServiceCenter {
  final String name;
  final double lat;
  final double lng;
  final String address;
  final String phone;

  ServiceCenter({
    required this.name,
    required this.lat,
    required this.lng,
    required this.address,
    required this.phone,
  });
}
