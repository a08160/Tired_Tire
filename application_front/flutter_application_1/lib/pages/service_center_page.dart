import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_map/flutter_map.dart';
import 'package:geolocator/geolocator.dart';
import 'package:latlong2/latlong.dart';
import 'dart:convert';
import 'package:csv/csv.dart';
import 'dart:math';

class ServiceCenterPage extends StatefulWidget {
  @override
  _LocationCircleMapPageState createState() => _LocationCircleMapPageState();
}

class _LocationCircleMapPageState extends State<ServiceCenterPage> {
  LatLng? _currentLocation;
  List<Map<String, dynamic>> _nearestCenters = [];

  @override
  void initState() {
    super.initState();
    _determinePosition();
  }

  Future<void> _determinePosition() async {
    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) return;

    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied ||
        permission == LocationPermission.deniedForever) {
      permission = await Geolocator.requestPermission();
      if (permission != LocationPermission.always &&
          permission != LocationPermission.whileInUse)
        return;
    }

    Position position = await Geolocator.getCurrentPosition();
    setState(() {
      _currentLocation = LatLng(position.latitude, position.longitude);
    });

    await _loadServiceCenters();
  }

  Future<void> _loadServiceCenters() async {
    final csvString = await rootBundle.loadString('assets/service_center.csv');
    final csvList = CsvToListConverter().convert(csvString, eol: '\n');

    final headers = csvList[0];
    final dataRows = csvList.sublist(1);

    int latIndex = headers.indexOf('위도');
    int lngIndex = headers.indexOf('경도');
    int nameIndex = headers.indexOf('자동차정비업체명');
    int addrIndex = headers.indexOf('소재지도로명주소');
    int typeIndex = headers.indexOf('자동차정비업체종류');

    List<Map<String, dynamic>> validCenters = [];

    for (var row in dataRows) {
      try {
        double lat = double.parse(row[latIndex].toString());
        double lng = double.parse(row[lngIndex].toString());
        double distance = Geolocator.distanceBetween(
          _currentLocation!.latitude,
          _currentLocation!.longitude,
          lat,
          lng,
        );
        validCenters.add({
          'name': row[nameIndex],
          'address': row[addrIndex],
          'type': row[typeIndex],
          'latlng': LatLng(lat, lng),
          'distance': distance,
        });
      } catch (_) {}
    }

    validCenters.sort((a, b) => a['distance'].compareTo(b['distance']));
    setState(() {
      _nearestCenters = validCenters.take(5).toList();
    });
  }

  String _formatDistance(double meters) {
    if (meters < 1000) {
      return '${meters.round()}m';
    } else {
      return '${(meters / 1000).toStringAsFixed(1)}km';
    }
  }

  String _mapType(dynamic code) {
    switch (code.toString()) {
      case '1':
        return '자동차 종합';
      case '2':
        return '소형 자동차 종합';
      case '3':
        return '자동차 전문';
      case '4':
        return '원동기 전문';
      case '99':
        return '기타';
      default:
        return '정보 없음';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("내 주변 정비소 찾기")),
      body:
          _currentLocation == null
              ? Center(child: CircularProgressIndicator())
              : Column(
                children: [
                  Expanded(
                    flex: 3,
                    child: FlutterMap(
                      options: MapOptions(
                        initialCenter: _currentLocation!,
                        initialZoom: 15.0,
                      ),
                      children: [
                        TileLayer(
                          urlTemplate:
                              'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                          subdomains: ['a', 'b', 'c'],
                        ),
                        MarkerLayer(
                          markers:
                              [
                                Marker(
                                  point: _currentLocation!,
                                  width: 60,
                                  height: 60,
                                  child: Container(
                                    decoration: BoxDecoration(
                                      shape: BoxShape.circle,
                                      color: Colors.blue.withOpacity(0.4),
                                      border: Border.all(
                                        color: Colors.blue,
                                        width: 2,
                                      ),
                                    ),
                                    child: Center(
                                      child: Container(
                                        width: 10,
                                        height: 10,
                                        decoration: BoxDecoration(
                                          color: Colors.blue,
                                          shape: BoxShape.circle,
                                        ),
                                      ),
                                    ),
                                  ),
                                ),
                              ] +
                              _nearestCenters
                                  .map(
                                    (center) => Marker(
                                      point: center['latlng'],
                                      width: 40,
                                      height: 40,
                                      child: Icon(
                                        Icons.location_on,
                                        color: Colors.red,
                                        size: 40,
                                      ),
                                    ),
                                  )
                                  .toList(),
                        ),
                      ],
                    ),
                  ),
                  Expanded(
                    flex: 2,
                    child: ListView.builder(
                      itemCount: _nearestCenters.length,
                      itemBuilder: (context, index) {
                        final c = _nearestCenters[index];
                        return ListTile(
                          title: Text(
                            c['name'],
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          subtitle: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(c['address'] ?? ''),
                              Text('거리: ${_formatDistance(c['distance'])}'),
                              Text('종류: ${_mapType(c['type'])}'),
                            ],
                          ),
                        );
                      },
                    ),
                  ),
                ],
              ),
    );
  }
}
